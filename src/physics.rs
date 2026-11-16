//! Native `PhysX` collision-cache access and deterministic static mesh cooking.
// Format references: NVIDIA PhysX 3.4/4.1 TriangleMeshBuilder::save and
// Gu::BV4Tree::load. Small BV4 meshes (<16 faces) use the SDK's tree-free path.
use crate::{codec, schema::RedSchema, world::WorldError};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

fn invalid(message: impl Into<String>) -> WorldError {
    WorldError::Invalid(message.into())
}
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], WorldError> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| invalid("truncated cooked collision"))?;
        let value = &self.bytes[self.at..end];
        self.at = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, WorldError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| invalid("cooked integer"))?,
        ))
    }
    fn f32(&mut self) -> Result<f32, WorldError> {
        let value = f32::from_bits(self.u32()?);
        if !value.is_finite() {
            return Err(invalid("nonfinite cooked vertex"));
        }
        Ok(value)
    }
    fn vector(&mut self) -> Result<[f32; 3], WorldError> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }
    fn header(&mut self, tag: &[u8]) -> Result<u32, WorldError> {
        let h = self.take(8)?;
        if &h[..3] != b"NXS" && &h[..3] != b"ICE" || h[3] & 1 != 1 || &h[4..] != tag {
            return Err(invalid(format!(
                "unsupported cooked collision header {h:?}"
            )));
        }
        self.u32()
    }
    fn count(&mut self) -> Result<usize, WorldError> {
        let count = usize::try_from(self.u32()?).map_err(|_| invalid("cooked count overflow"))?;
        if count > 8_000_000 {
            return Err(invalid("cooked count exceeds limit"));
        }
        Ok(count)
    }
}

/// Rigid collision surfaces in native shape-local coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionGeometry {
    /// True for a solid convex hull rather than a triangle surface resource.
    #[serde(default)]
    pub convex: bool,
    /// Native XYZ vertices.
    pub vertices: Vec<[f32; 3]>,
    /// Outward native triangle winding.
    pub triangles: Vec<[u32; 3]>,
    /// Per-face material slot; empty means slot zero.
    pub materials: Vec<u16>,
}

/// A native PhysX heightfield sample, including hole materials and its diagonal flag.
pub struct HeightFieldSample {
    /// Signed native height before applying the node's height scale.
    pub height: i16,
    /// Two seven-bit material indices; bit seven of the first selects the diagonal.
    pub materials: [u8; 2],
}
/// Row-major native heightfield data. Node transforms supply its physical scales.
pub struct HeightField {
    /// Native grid dimensions, including boundary vertices.
    pub rows: usize,
    /// Number of columns in each sample row.
    pub columns: usize,
    /// All row-major height samples and material flags.
    pub samples: Vec<HeightFieldSample>,
}
/// Reads PhysX HFHF cooked heightfields, including native tessellation bits.
/// Layout: PhysX `GuHeightField.cpp`, `HeightField::load`.
///
/// # Errors
/// Rejects unsupported versions, formats, strides and inconsistent sample counts.
pub fn decode_heightfield(bytes: &[u8]) -> Result<HeightField, WorldError> {
    let mut r = Reader { bytes, at: 0 };
    let version = r.header(b"HFHF")?;
    if version != 1 {
        return Err(invalid(format!("heightfield version {version}")));
    }
    let rows = r.count()?;
    let columns = r.count()?;
    for _ in 0..5 {
        r.f32()?;
    }
    r.take(2)?;
    if r.u32()? != 1 {
        return Err(invalid("heightfield sample format"));
    }
    for _ in 0..6 {
        r.f32()?;
    }
    let stride = r.u32()?;
    let count = r.count()?;
    r.f32()?;
    r.f32()?;
    if rows < 2 || columns < 2 || rows.checked_mul(columns) != Some(count) || stride != 4 {
        return Err(invalid("heightfield sample dimensions"));
    }
    let samples = (0..count)
        .map(|_| {
            let raw = r.take(4)?;
            Ok(HeightFieldSample {
                height: i16::from_le_bytes([raw[0], raw[1]]),
                materials: [raw[2], raw[3]],
            })
        })
        .collect::<Result<Vec<_>, WorldError>>()?;
    if r.at != bytes.len() {
        return Err(invalid("heightfield trailing bytes"));
    }
    Ok(HeightField {
        rows,
        columns,
        samples,
    })
}

/// Decodes one cooked triangle or convex mesh and returns its exact byte span.
///
/// # Errors
/// Rejects malformed counts, indices and unsupported cooked stream versions.
#[allow(clippy::too_many_lines, clippy::float_cmp)] // Exact binary float sentinels, shared stream cursor.
pub fn decode_cooked(bytes: &[u8]) -> Result<(CollisionGeometry, usize), WorldError> {
    let mut r = Reader { bytes, at: 0 };
    let tag = bytes
        .get(4..8)
        .ok_or_else(|| invalid("cooked header absent"))?;
    let version = r.header(tag)?;
    let mut geometry = CollisionGeometry {
        convex: tag == b"CVXM",
        vertices: Vec::new(),
        triangles: Vec::new(),
        materials: Vec::new(),
    };
    if tag == b"MESH" {
        if version != 15 {
            return Err(invalid(format!("triangle collision version {version}")));
        }
        if r.u32()? != 1 {
            return Err(invalid("collision uses a non-BV4 midphase"));
        }
        let flags = r.u32()?;
        let nv = r.count()?;
        let nt = r.count()?;
        for _ in 0..nv {
            geometry.vertices.push(r.vector()?);
        }
        let width = if flags & 4 != 0 {
            1
        } else if flags & 8 != 0 {
            2
        } else {
            4
        };
        for _ in 0..nt {
            let mut t = [0; 3];
            for i in &mut t {
                *i = match width {
                    1 => u32::from(r.take(1)?[0]),
                    2 => u32::from(u16::from_le_bytes(
                        r.take(2)?.try_into().map_err(|_| invalid("index"))?,
                    )),
                    _ => r.u32()?,
                };
                if *i as usize >= nv {
                    return Err(invalid("collision triangle index outside vertices"));
                }
            }
            geometry.triangles.push(t);
        }
        if flags & 1 != 0 {
            for _ in 0..nt {
                geometry.materials.push(u16::from_le_bytes(
                    r.take(2)?
                        .try_into()
                        .map_err(|_| invalid("material index"))?,
                ));
            }
        }
        if flags & 2 != 0 {
            let max = r.u32()?;
            let size = if max <= 255 {
                1
            } else if max <= 65_535 {
                2
            } else {
                4
            };
            r.take(nt * size)?;
        }
        if flags & 16 != 0 {
            r.take(nt * 12)?;
        }
        if r.take(4)? != b"BV4 " {
            return Err(invalid("collision BV4 tree absent"));
        }
        let tree_version = r.u32()?;
        if !(1..=3).contains(&tree_version) {
            return Err(invalid("collision BV4 version"));
        }
        r.take(44)?;
        let quantized = tree_version < 3 || r.u32()? != 0;
        let nodes = r.count()?;
        r.take(nodes * if quantized { 16 } else { 28 })?;
        r.take(28)?;
        let extra = r.count()?;
        r.take(extra)?;
        if flags & 32 != 0 {
            r.take(nt * (3 * width + 16 + 4))?;
            if r.take(4)? != b"BV32" {
                return Err(invalid("collision GPU BV32 tree absent"));
            }
            let v = r.u32()?;
            if v != 1 {
                return Err(invalid("collision GPU BV32 version"));
            }
            r.take(20)?;
            let blocks = r.count()?;
            for _ in 0..blocks {
                let count = r.count()?;
                r.take(count * 4 + count * 16 * 2)?;
            }
        }
        if flags & !63 != 0 {
            return Err(invalid(format!("unknown cooked mesh flags {flags}")));
        }
    } else if tag == b"CVXM" {
        if version != 13 {
            return Err(invalid(format!("convex collision version {version}")));
        }
        r.u32()?;
        let hull_version = r.header(b"CLHL")?;
        if hull_version <= 8 {
            r.header(b"CVHL")?;
        }
        let nv = r.count()?;
        let edges = r.count()?;
        let np = r.count()?;
        let ni = r.count()?;
        if nv > 255 || np > 255 {
            return Err(invalid("convex topology exceeds native byte indices"));
        }
        for _ in 0..nv {
            geometry.vertices.push(r.vector()?);
        }
        let mut faces = Vec::new();
        for _ in 0..np {
            r.take(16)?;
            let at = u16::from_le_bytes(
                r.take(2)?
                    .try_into()
                    .map_err(|_| invalid("convex face offset"))?,
            ) as usize;
            let n = usize::from(r.take(1)?[0]);
            r.take(1)?;
            faces.push((at, n));
        }
        let indices = r.take(ni)?;
        for (at, n) in faces {
            let vertices = indices
                .get(at..at + n)
                .ok_or_else(|| invalid("convex polygon outside index table"))?;
            if vertices.iter().any(|v| usize::from(*v) >= nv) {
                return Err(invalid("convex vertex index"));
            }
            for i in 1..n.saturating_sub(1) {
                geometry.triangles.push([
                    u32::from(vertices[0]),
                    u32::from(vertices[i]),
                    u32::from(vertices[i + 1]),
                ]);
            }
        }
        r.take((edges & 0x7fff) * 2 + nv * 3)?;
        if edges & 0x8000 != 0 {
            r.take((edges & 0x7fff) * 4)?;
        }
        r.take(28)?;
        let mass = r.f32()?;
        if mass != -1. {
            r.take(48)?;
        }
        let gauss = r.f32()?;
        if gauss != -1. {
            if gauss != 1. {
                return Err(invalid("convex Gauss map flag"));
            }
            r.header(b"SUPM")?;
            r.header(b"GAUS")?;
            r.u32()?;
            let samples = r.count()?;
            r.take(samples * 2)?;
            r.header(b"VALE")?;
            let vertices = r.count()?;
            let adjacent = r.count()?;
            let max = r.u32()?;
            r.take(vertices * if max <= 255 { 1 } else { 2 })?;
            r.take(adjacent)?;
        }
        r.take(16)?;
    } else {
        return Err(invalid(format!("unsupported collision tag {tag:?}")));
    }
    Ok((geometry, r.at))
}

fn clip(polygon: &[Vec3], plane: [f32; 4], inside: bool) -> Vec<Vec3> {
    let mut result = Vec::new();
    let Some(&last) = polygon.last() else {
        return result;
    };
    let distance = |p: Vec3| Vec3::new(plane[0], plane[1], plane[2]).dot(p) + plane[3];
    let mut previous = last;
    let mut pd = distance(previous);
    for &point in polygon {
        let d = distance(point);
        let keep = if inside { d <= 0. } else { d >= 0. };
        let was = if inside { pd <= 0. } else { pd >= 0. };
        if keep != was {
            result.push(previous.lerp(point, (pd / (pd - d)).clamp(0., 1.)));
        }
        if keep {
            result.push(point);
        }
        previous = point;
        pd = d;
    }
    result
}

impl CollisionGeometry {
    /// Constructs an outward-wound solid box for exact non-axis-aligned cuts.
    #[must_use]
    pub fn solid_box(min: [f32; 3], max: [f32; 3]) -> Self {
        let vertices = [
            (min[0], min[1], min[2]),
            (max[0], min[1], min[2]),
            (max[0], max[1], min[2]),
            (min[0], max[1], min[2]),
            (min[0], min[1], max[2]),
            (max[0], min[1], max[2]),
            (max[0], max[1], max[2]),
            (min[0], max[1], max[2]),
        ]
        .map(|(x, y, z)| [x, y, z])
        .to_vec();
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        Self {
            convex: true,
            vertices,
            triangles,
            materials: vec![0; 12],
        }
    }

    fn half_hull(
        faces: &[(Vec<Vec3>, u16)],
        plane: [f32; 4],
        inside: bool,
    ) -> Vec<(Vec<Vec3>, u16)> {
        let mut output = Vec::new();
        let mut cuts = Vec::<Vec3>::new();
        let normal_vector = Vec3::new(plane[0], plane[1], plane[2]);
        let distances: Vec<_> = faces
            .iter()
            .flat_map(|(face, _)| face)
            .map(|p| normal_vector.dot(*p) + plane[3])
            .collect();
        let crosses = distances.iter().any(|d| *d > 1e-6) && distances.iter().any(|d| *d < -1e-6);
        for (face, material) in faces {
            let polygon = clip(face, plane, inside);
            if polygon.len() >= 3 {
                output.push((polygon, *material));
            }
            for i in 0..face.len() {
                let a = face[i];
                let b = face[(i + 1) % face.len()];
                let da = normal_vector.dot(a) + plane[3];
                let db = normal_vector.dot(b) + plane[3];
                if da * db < 0. {
                    let p = a.lerp(b, da / (da - db));
                    if !cuts.iter().any(|v| v.distance_squared(p) < 1e-10) {
                        cuts.push(p);
                    }
                } else if da.abs() < 1e-6 && !cuts.iter().any(|v| v.distance_squared(a) < 1e-10) {
                    cuts.push(a);
                }
            }
        }
        if crosses && cuts.len() >= 3 {
            #[allow(clippy::cast_precision_loss)]
            // Centroid divisor; native input counts are bounded.
            let divisor = cuts.len() as f32;
            let center = cuts.iter().copied().sum::<Vec3>() / divisor;
            let normal = if inside {
                normal_vector.normalize()
            } else {
                -normal_vector.normalize()
            };
            let axis = if normal.x.abs() < 0.8 {
                Vec3::X
            } else {
                Vec3::Y
            };
            let u = normal.cross(axis).normalize();
            let v = normal.cross(u);
            cuts.sort_by(|a, b| {
                let a = *a - center;
                let b = *b - center;
                a.dot(v)
                    .atan2(a.dot(u))
                    .total_cmp(&b.dot(v).atan2(b.dot(u)))
            });
            output.push((cuts, faces.first().map_or(0, |f| f.1)));
        }
        output
    }

    /// Subtracts a convex brush from a solid hull, including closed aperture reveals.
    ///
    /// # Errors
    /// Rejects invalid topology or cut planes.
    pub fn subtract_solid(&self, planes: &[[f32; 4]]) -> Result<Self, WorldError> {
        if !self.convex {
            return self.subtract(planes);
        }
        if planes.is_empty()
            || planes.iter().any(|p| {
                p.iter().any(|v| !v.is_finite())
                    || Vec3::new(p[0], p[1], p[2]).length_squared() < 1e-12
            })
        {
            return Err(invalid("invalid solid cut planes"));
        }
        let mut remaining = Vec::new();
        for (i, t) in self.triangles.iter().enumerate() {
            let face = t
                .iter()
                .map(|&v| {
                    self.vertices
                        .get(v as usize)
                        .copied()
                        .map(Vec3::from_array)
                        .ok_or_else(|| invalid("solid collision vertex index"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            remaining.push((face, self.materials.get(i).copied().unwrap_or(0)));
        }
        let mut out = Self {
            convex: false,
            vertices: Vec::new(),
            triangles: Vec::new(),
            materials: Vec::new(),
        };
        for &plane in planes {
            let outside = Self::half_hull(&remaining, plane, false);
            for (face, material) in outside {
                for i in 1..face.len().saturating_sub(1) {
                    let p = [face[0], face[i], face[i + 1]];
                    if (p[1] - p[0]).cross(p[2] - p[0]).length_squared() < 1e-12 {
                        continue;
                    }
                    let first = u32::try_from(out.vertices.len())
                        .map_err(|_| invalid("solid output index overflow"))?;
                    out.vertices.extend(p.map(|v| v.to_array()));
                    out.triangles.push([first, first + 1, first + 2]);
                    out.materials.push(material);
                }
            }
            remaining = Self::half_hull(&remaining, plane, true);
            if remaining.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    /// Subtracts a convex volume while retaining source face material slots.
    ///
    /// # Errors
    /// Rejects nonfinite planes or invalid source topology.
    pub fn subtract(&self, planes: &[[f32; 4]]) -> Result<Self, WorldError> {
        if planes.is_empty() || planes.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
            return Err(invalid("invalid collision cut planes"));
        }
        let mut out = Self {
            convex: false,
            vertices: Vec::new(),
            triangles: Vec::new(),
            materials: Vec::new(),
        };
        for (face, t) in self.triangles.iter().enumerate() {
            let mut remaining = t
                .iter()
                .map(|&i| {
                    self.vertices
                        .get(i as usize)
                        .copied()
                        .map(Vec3::from_array)
                        .ok_or_else(|| invalid("collision vertex index"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            for &plane in planes {
                if remaining.is_empty() {
                    break;
                }
                let outside = clip(&remaining, plane, false);
                for i in 1..outside.len().saturating_sub(1) {
                    let points = [outside[0], outside[i], outside[i + 1]];
                    if (points[1] - points[0])
                        .cross(points[2] - points[0])
                        .length_squared()
                        < 1e-12
                    {
                        continue;
                    }
                    let first = u32::try_from(out.vertices.len())
                        .map_err(|_| invalid("collision output exceeds index range"))?;
                    out.vertices.extend(points.map(|p| p.to_array()));
                    out.triangles.push([first, first + 1, first + 2]);
                    out.materials
                        .push(self.materials.get(face).copied().unwrap_or(0));
                }
                remaining = clip(&remaining, plane, true);
            }
        }
        Ok(out)
    }

    /// Cooks exact static triangle surfaces into native small-BV4 streams.
    /// Each stream uses the SDK's documented tree-free path for fewer than 16 faces.
    ///
    /// # Errors
    /// Rejects malformed or nonfinite geometry.
    pub fn cook(&self) -> Result<Vec<Vec<u8>>, WorldError> {
        let mut streams = Vec::new();
        for (chunk, triangles) in self.triangles.chunks(15).enumerate() {
            let mut vertices = Vec::new();
            for t in triangles {
                for &i in t {
                    let p = *self
                        .vertices
                        .get(i as usize)
                        .ok_or_else(|| invalid("collision cook index"))?;
                    if p.iter().any(|v| !v.is_finite()) {
                        return Err(invalid("nonfinite collision cook vertex"));
                    }
                    vertices.push(p);
                }
            }
            let mut bytes = b"NXS\x01MESH".to_vec();
            let u = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
            let f = |out: &mut Vec<u8>, v: f32| out.extend(v.to_le_bytes());
            for v in [
                15,
                1,
                5,
                u32::try_from(vertices.len()).map_err(|_| invalid("vertex count overflow"))?,
                u32::try_from(triangles.len()).map_err(|_| invalid("face count overflow"))?,
            ] {
                u(&mut bytes, v);
            }
            for p in &vertices {
                for &v in p {
                    f(&mut bytes, v);
                }
            }
            bytes.extend(
                0..u8::try_from(vertices.len())
                    .map_err(|_| invalid("small collision mesh vertex overflow"))?,
            );
            for i in 0..triangles.len() {
                bytes.extend(
                    self.materials
                        .get(chunk * 15 + i)
                        .copied()
                        .unwrap_or(0)
                        .to_le_bytes(),
                );
            }
            let mut low = Vec3::splat(f32::INFINITY);
            let mut high = Vec3::splat(f32::NEG_INFINITY);
            for &p in &vertices {
                low = low.min(Vec3::from_array(p));
                high = high.max(Vec3::from_array(p));
            }
            bytes.extend(b"BV4 ");
            u(&mut bytes, 2);
            for v in ((low + high) / 2.).to_array() {
                f(&mut bytes, v);
            }
            f(&mut bytes, (high - low).length() / 2.);
            u(&mut bytes, 0);
            for _ in 0..6 {
                f(&mut bytes, 1.);
            }
            u(&mut bytes, 0);
            f(&mut bytes, 0.0001);
            for v in low.to_array().into_iter().chain(high.to_array()) {
                f(&mut bytes, v);
            }
            u(&mut bytes, 0);
            let (decoded, consumed) = decode_cooked(&bytes)?;
            if consumed != bytes.len() || decoded.vertices != vertices {
                return Err(invalid("cooked collision readback differs"));
            }
            streams.push(bytes);
        }
        Ok(streams)
    }
}

#[derive(Debug)]
pub(crate) struct GeometryCache {
    bytes: Vec<u8>,
    entries: HashMap<(u64, u64), (usize, usize)>,
    buffers: Mutex<HashMap<usize, Vec<Arc<CollisionGeometry>>>>,
}
impl GeometryCache {
    pub(crate) fn new(bytes: Vec<u8>, schema: &RedSchema) -> Result<Self, WorldError> {
        let document = codec::decode_metadata_from_bytes(&bytes, schema)
            .map_err(|e| invalid(e.to_string()))?;
        let root = &document["Data"]["RootChunk"];
        let mut entries = HashMap::new();
        let sectors = root["sectorEntries"]
            .as_array()
            .ok_or_else(|| invalid("collision cache sectors absent"))?;
        let buffers = root["bufferTableSectors"]
            .as_array()
            .ok_or_else(|| invalid("collision cache buffers absent"))?;
        let keys = root["sectorGeometries"]
            .as_array()
            .ok_or_else(|| invalid("collision cache keys absent"))?;
        let number = |v: &Value| {
            if v.is_null() {
                return Ok(0);
            }
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                .ok_or_else(|| invalid(format!("collision cache integer {v}")))
        };
        for (index, sector) in sectors.iter().enumerate().chain(std::iter::once((
            sectors.len(),
            &root["alwaysLoadedSector"],
        ))) {
            let hash = number(&sector["sectorHash"])?;
            let offset = usize::try_from(number(&sector["entryOffset"])?)
                .map_err(|_| invalid("sector offset overflow"))?;
            let count = usize::try_from(number(&sector["entrySize"])?)
                .map_err(|_| invalid("sector count overflow"))?;
            if count == 0 {
                continue;
            }
            let buffer = if index == sectors.len() {
                &root["alwaysLoadedSectorDDB"]
            } else {
                buffers
                    .get(index)
                    .ok_or_else(|| invalid("collision cache sector buffer"))?
            };
            let buffer = buffer["BufferId"]
                .as_str()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| invalid("collision cache buffer identity"))?;
            for local in 0..count {
                let key = keys
                    .get(offset + local)
                    .ok_or_else(|| invalid("collision cache key offset"))?;
                let data = key["ta"]
                    .as_array()
                    .or_else(|| key["ta"]["Elements"].as_array())
                    .ok_or_else(|| invalid("collision cache hash bytes"))?;
                let mut raw = [0_u8; 8];
                for (i, v) in raw.iter_mut().enumerate() {
                    *v = u8::try_from(number(
                        data.get(i)
                            .ok_or_else(|| invalid("collision cache hash truncated"))?,
                    )?)
                    .map_err(|_| invalid("collision cache hash byte"))?;
                }
                entries.insert((hash, u64::from_le_bytes(raw)), (buffer, local));
            }
        }
        Ok(Self {
            bytes,
            entries,
            buffers: Mutex::new(HashMap::new()),
        })
    }
    pub(crate) fn mesh(
        &self,
        sector: u64,
        geometry: u64,
    ) -> Result<Arc<CollisionGeometry>, WorldError> {
        let &(buffer, index) = self
            .entries
            .get(&(sector, geometry))
            .or_else(|| self.entries.get(&(0, geometry)))
            .ok_or_else(|| invalid(format!("cooked collision {sector}/{geometry} absent")))?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| invalid("collision cache lock poisoned"))?;
        if let std::collections::hash_map::Entry::Vacant(entry) = buffers.entry(buffer) {
            let raw = codec::decode_table_buffer(&self.bytes, buffer)
                .map_err(|e| invalid(e.to_string()))?;
            let mut at = 0;
            let mut meshes = Vec::new();
            while at < raw.len() {
                let (mesh, n) = decode_cooked(&raw[at..])?;
                if n == 0 {
                    return Err(invalid("empty collision cache entry"));
                }
                at += n;
                meshes.push(Arc::new(mesh));
            }
            entry.insert(meshes);
        }
        buffers[&buffer]
            .get(index)
            .cloned()
            .ok_or_else(|| invalid("collision cache entry count differs"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ray(mesh: &CollisionGeometry, origin: Vec3, direction: Vec3) -> bool {
        mesh.triangles.iter().any(|t| {
            let [a, b, c] = t.map(|i| Vec3::from_array(mesh.vertices[i as usize]));
            let e1 = b - a;
            let e2 = c - a;
            let p = direction.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-8 {
                return false;
            }
            let inv = 1. / det;
            let s = origin - a;
            let u = s.dot(p) * inv;
            if !(0. ..=1.).contains(&u) {
                return false;
            }
            let q = s.cross(e1);
            let v = direction.dot(q) * inv;
            v >= 0. && u + v <= 1. && e2.dot(q) * inv > 0.
        })
    }
    #[test]
    fn solid_aperture_preserves_volume_and_closes_reveals() {
        let wall = CollisionGeometry::solid_box([-3., -0.1, 0.], [3., 0.1, 4.]);
        let planes = [
            [1., 0., 0., -0.75],
            [-1., 0., 0., -0.75],
            [0., 1., 0., -1.],
            [0., -1., 0., -1.],
            [0., 0., 1., -2.5],
            [0., 0., -1., -0.1],
        ];
        let cut = wall.subtract_solid(&planes).unwrap();
        let volume: f32 = cut
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| Vec3::from_array(cut.vertices[i as usize]));
                a.dot(b.cross(c)) / 6.
            })
            .sum();
        assert!((volume - (6. * 0.2 * 4. - 1.5 * 0.2 * 2.5)).abs() < 0.0001);
        assert!(!ray(&cut, Vec3::new(0., -2., 1.), Vec3::Y));
        assert!(ray(&cut, Vec3::new(1.5, -2., 1.), Vec3::Y));
        assert!(ray(&cut, Vec3::new(0., -2., 3.), Vec3::Y));
        assert!(ray(&cut, Vec3::new(0., 0., 1.), Vec3::X));
    }
    #[test]
    fn native_small_bv4_streams_preserve_surface_topology_and_materials() {
        let source = CollisionGeometry::solid_box([-1.; 3], [1.; 3]);
        let planes = [[1., 0., 0., 0.], [-1., 0., 0., -0.5]];
        let cut = source.subtract_solid(&planes).unwrap();
        let streams = cut.cook().unwrap();
        let mut triangles = 0;
        for stream in streams {
            let (mesh, bytes) = decode_cooked(&stream).unwrap();
            assert_eq!(bytes, stream.len());
            assert!(mesh.triangles.len() < 16);
            triangles += mesh.triangles.len();
            assert!(mesh.materials.iter().all(|m| *m == 0));
        }
        assert_eq!(triangles, cut.triangles.len());
    }
}
