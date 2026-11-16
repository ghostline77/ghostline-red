//! Topology-preserving native mesh surface cuts with interpolated vertex streams.
use super::*;
use glam::Vec3;
use std::collections::HashMap;

/// Counts from a native surface cut, including every source LOD.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CutReport {
    /// Source triangle count.
    pub triangles_before: usize,
    /// Retained/retriangulated triangle count.
    pub triangles_after: usize,
    /// Number of source triangles intersected or removed.
    pub affected_triangles: usize,
    /// Native render chunks after vertex-limit splitting.
    pub chunks: usize,
}

#[derive(Clone, Copy)]
struct Point {
    position: Vec3,
    weights: [f32; 3],
}

fn clip(polygon: &[Point], plane: [f32; 4], inside: bool) -> Vec<Point> {
    let mut output = Vec::new();
    if polygon.is_empty() {
        return output;
    }
    let distance = |p: Point| Vec3::new(plane[0], plane[1], plane[2]).dot(p.position) + plane[3];
    let mut previous = polygon[polygon.len() - 1];
    let mut previous_d = distance(previous);
    for &current in polygon {
        let current_d = distance(current);
        let keep = if inside {
            current_d <= 0.000_001
        } else {
            current_d >= 0.000_001
        };
        let previous_keep = if inside {
            previous_d <= 0.000_001
        } else {
            previous_d >= 0.000_001
        };
        if keep != previous_keep {
            let t = (previous_d / (previous_d - current_d)).clamp(0., 1.);
            output.push(Point {
                position: previous.position.lerp(current.position, t),
                weights: std::array::from_fn(|i| {
                    previous.weights[i] + (current.weights[i] - previous.weights[i]) * t
                }),
            });
        }
        if keep {
            output.push(current);
        }
        previous = current;
        previous_d = current_d;
    }
    output
}

struct Chunk {
    original: usize,
    vertices: Vec<([u16; 3], [f32; 3])>,
    indices: Vec<u16>,
    lookup: HashMap<Vec<(u16, i32)>, u16>,
}
impl Chunk {
    fn new(original: usize) -> Self {
        Self {
            original,
            vertices: Vec::new(),
            indices: Vec::new(),
            lookup: HashMap::new(),
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "weights are finite and bounded to the unit interval"
    )]
    fn vertex(&mut self, triangle: [u16; 3], point: Point) -> Result<u16, MeshError> {
        let mut key = triangle
            .into_iter()
            .zip(point.weights)
            .filter(|(_, w)| *w > 0.000_001)
            .map(|(i, w)| (i, (w * 1_000_000.).round() as i32))
            .collect::<Vec<_>>();
        key.sort_unstable();
        if let Some(index) = self.lookup.get(&key) {
            return Ok(*index);
        }
        let index = u16::try_from(self.vertices.len())
            .map_err(|_| malformed("cut chunk exceeds native vertex index range"))?;
        self.vertices.push((triangle, point.weights));
        self.lookup.insert(key, index);
        Ok(index)
    }
}

/// Subtracts a convex volume from rigid native mesh surfaces, preserving all LODs,
/// material bindings, UVs, vertex colours and interpolated normal/tangent streams.
/// Planes use decoded glTF coordinates and keep `normal.dot(position)+offset <= 0`
/// inside the removed volume. Aperture reveals can be authored as separate lining.
/// The source document is changed only after the full new native buffer is valid.
///
/// # Errors
/// Rejects invalid planes, skinned/deformed meshes, malformed streams and cuts
/// that remove all geometry. World entities should be edited at their rigid mesh
/// components, not by cutting an animated character or door skeleton.
#[expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "native packed-stream encoding validates ranges before bounded numeric conversion"
)]
pub fn subtract_convex_volume(
    document: &mut Value,
    planes: &[[f32; 4]],
) -> Result<CutReport, MeshError> {
    if planes.is_empty()
        || planes.len() > 32
        || planes.iter().any(|p| {
            p.iter().any(|n| !n.is_finite()) || Vec3::new(p[0], p[1], p[2]).length() < 0.000_001
        })
    {
        return Err(malformed("invalid convex cut planes"));
    }
    let meshes = decode_geometry(document, false)?;
    if meshes.iter().any(|m| !m.skin.is_empty()) {
        return Err(malformed(
            "surface cuts require rigid native mesh components",
        ));
    }
    let root = &document["Data"]["RootChunk"];
    let header = &root["renderResourceBlob"]["Data"]["header"];
    let originals = array_member(header, "renderChunkInfos")?;
    let source_bytes = STANDARD.decode(string_member(
        &root["renderResourceBlob"]["Data"]["renderBuffer"],
        "Bytes",
    )?)?;
    let scale = quantization_vector(member(header, "quantizationScale")?)?;
    let offset = quantization_vector(member(header, "quantizationOffset")?)?;
    let mut report = CutReport {
        triangles_before: 0,
        triangles_after: 0,
        affected_triangles: 0,
        chunks: 0,
    };
    let mut chunks = Vec::new();
    for mesh in &meshes {
        let mut output = Chunk::new(mesh.chunk_index);
        for triangle in mesh.indices.chunks_exact(3) {
            report.triangles_before += 1;
            let indices = [triangle[0], triangle[1], triangle[2]];
            let polygon = indices
                .iter()
                .enumerate()
                .map(|(i, &index)| Point {
                    position: Vec3::from_array(mesh.positions[usize::from(index)]),
                    weights: std::array::from_fn(|j| if i == j { 1. } else { 0. }),
                })
                .collect::<Vec<_>>();
            let mut remaining = polygon.clone();
            let mut outside = Vec::new();
            for &plane in planes {
                if remaining.is_empty() {
                    break;
                }
                let escaped = clip(&remaining, plane, false);
                if escaped.len() >= 3 {
                    outside.push(escaped);
                }
                remaining = clip(&remaining, plane, true);
            }
            if !remaining.is_empty() {
                report.affected_triangles += 1;
            }
            for polygon in outside {
                for i in 1..polygon.len() - 1 {
                    let points = [polygon[0], polygon[i], polygon[i + 1]];
                    if (points[1].position - points[0].position)
                        .cross(points[2].position - points[0].position)
                        .length_squared()
                        < 1e-14
                    {
                        continue;
                    }
                    if output.vertices.len() > 65_500 {
                        chunks.push(output);
                        output = Chunk::new(mesh.chunk_index);
                    }
                    let a = output.vertex(indices, points[0])?;
                    let b = output.vertex(indices, points[1])?;
                    let c = output.vertex(indices, points[2])?;
                    output.indices.extend_from_slice(&[a, b, c]);
                    report.triangles_after += 1;
                }
            }
        }
        if !output.indices.is_empty() {
            chunks.push(output);
        }
    }
    if chunks.is_empty() {
        return Err(malformed(
            "cut removes all geometry; remove the source placement instead",
        ));
    }
    let mut bytes = Vec::new();
    let mut indices_bytes = Vec::new();
    let mut headers = Vec::new();
    let mut topology = Vec::new();
    for chunk in &chunks {
        let mesh = meshes
            .iter()
            .find(|m| m.chunk_index == chunk.original)
            .ok_or_else(|| malformed("cut source chunk absent"))?;
        let native = &originals[chunk.original];
        let mut encoded = native.clone();
        let old_offsets = usize_array(member(&native["chunkVertices"], "byteOffsets")?)?;
        let strides = usize_array(member(
            &native["chunkVertices"]["vertexLayout"],
            "slotStrides",
        )?)?;
        let elements = array_member(&native["chunkVertices"]["vertexLayout"], "elements")?;
        let mut new_offsets = old_offsets.clone();
        for stream in 0..old_offsets.len() {
            let stride = strides.get(stream).copied().unwrap_or(0);
            if stride == 0 {
                continue;
            }
            while bytes.len() % 16 != 0 {
                bytes.push(0);
            }
            new_offsets[stream] = bytes.len();
            for &(triangle, weights) in &chunk.vertices {
                let nearest = (0..3)
                    .max_by(|&a, &b| weights[a].total_cmp(&weights[b]))
                    .unwrap_or(0);
                let start = old_offsets[stream] + usize::from(triangle[nearest]) * stride;
                bytes.extend_from_slice(
                    source_bytes
                        .get(start..start + stride)
                        .ok_or_else(|| malformed("cut source vertex stream truncated"))?,
                );
            }
        }
        let position = offset_for_usage(elements, &new_offsets, "PS_Position", 0)?
            .ok_or_else(|| malformed("position stream absent"))?;
        let normal = offset_for_usage(elements, &new_offsets, "PS_Normal", 0)?;
        let tangent = offset_for_usage(elements, &new_offsets, "PS_Tangent", 0)?;
        let uv0 = offset_for_usage(elements, &new_offsets, "PS_TexCoord", 0)?;
        let uv1 = offset_for_usage(elements, &new_offsets, "PS_TexCoord", 1)?;
        let color = offset_for_usage(elements, &new_offsets, "PS_Color", 0)?;
        for (vertex, &(triangle, weights)) in chunk.vertices.iter().enumerate() {
            let blend3 = |values: &[[f32; 3]]| -> Vec3 {
                triangle
                    .iter()
                    .zip(weights)
                    .map(|(&i, w)| Vec3::from_array(values[usize::from(i)]) * w)
                    .sum()
            };
            let point = blend3(&mesh.positions);
            let red = [point.x, -point.z, point.y];
            let at = position + vertex * strides[0];
            for axis in 0..3 {
                let q = (((red[axis] - offset[axis]) / scale[axis]).clamp(-1., 1.) * 32767.).round()
                    as i16;
                bytes[at + axis * 2..at + axis * 2 + 2].copy_from_slice(&q.to_le_bytes());
            }
            if let Some(start) = normal
                && !mesh.normals.is_empty()
            {
                let n = blend3(&mesh.normals).normalize_or_zero().to_array();
                let at = start + vertex * if tangent.is_some() { 8 } else { 4 };
                bytes[at..at + 4].copy_from_slice(&encode_direction(n, 1.)?.to_le_bytes());
            }
            if let Some(start) = tangent
                && !mesh.tangents.is_empty()
            {
                let mut v = [0.; 4];
                for (&i, w) in triangle.iter().zip(weights) {
                    for (axis, value) in v.iter_mut().enumerate() {
                        *value += mesh.tangents[usize::from(i)][axis] * w;
                    }
                }
                let n = Vec3::new(v[0], v[1], v[2]).normalize_or_zero().to_array();
                let at = start
                    + usize::from(normal.is_some()) * 4
                    + vertex * if normal.is_some() { 8 } else { 4 };
                bytes[at..at + 4].copy_from_slice(
                    &encode_direction(n, if v[3] < 0. { -1. } else { 1. })?.to_le_bytes(),
                );
            }
            for (start, values, stride, extra, flip) in [
                (uv0, &mesh.tex_coords, 4, 0, true),
                (
                    uv1,
                    &mesh.tex_coords_1,
                    if color.is_some() { 8 } else { 4 },
                    usize::from(color.is_some()) * 4,
                    false,
                ),
            ] {
                if let Some(start) = start
                    && !values.is_empty()
                {
                    let mut uv = [0.; 2];
                    for (&i, w) in triangle.iter().zip(weights) {
                        for axis in 0..2 {
                            uv[axis] += values[usize::from(i)][axis] * w;
                        }
                    }
                    if flip {
                        uv[1] = 1. - uv[1];
                    }
                    let at = start + extra + vertex * stride;
                    for axis in 0..2 {
                        bytes[at + axis * 2..at + axis * 2 + 2]
                            .copy_from_slice(&f16::from_f32(uv[axis]).to_bits().to_le_bytes());
                    }
                }
            }
            if let Some(start) = color
                && !mesh.colors.is_empty()
            {
                let mut v = [0.; 4];
                for (&i, w) in triangle.iter().zip(weights) {
                    for (axis, value) in v.iter_mut().enumerate() {
                        *value += mesh.colors[usize::from(i)][axis] * w;
                    }
                }
                let at = start + vertex * if uv1.is_some() { 8 } else { 4 };
                for axis in 0..4 {
                    bytes[at + axis] = (v[axis].clamp(0., 1.) * 255.).round() as u8;
                }
            }
        }
        encoded["chunkVertices"]["byteOffsets"] = json!(new_offsets);
        encoded["numVertices"] = json!(chunk.vertices.len());
        encoded["numIndices"] = json!(chunk.indices.len());
        while indices_bytes.len() % 16 != 0 {
            indices_bytes.push(0);
        }
        encoded["chunkIndices"]["teOffset"] = json!(indices_bytes.len());
        for triangle in chunk.indices.chunks_exact(3) {
            for index in [triangle[1], triangle[0], triangle[2]] {
                indices_bytes.extend_from_slice(&index.to_le_bytes());
            }
        }
        headers.push(encoded);
        topology.push(
            header["topology"]
                .get(chunk.original)
                .cloned()
                .unwrap_or_else(|| json!({"$type":"rendTopologyData"})),
        );
    }
    while bytes.len() % 16 != 0 {
        bytes.push(0);
    }
    let vertex_size = bytes.len();
    let index_size = indices_bytes.len();
    bytes.extend(indices_bytes);
    let mut edited = document.clone();
    let header = &mut edited["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["header"];
    header["renderChunkInfos"] = json!(headers);
    header["topology"] = json!(topology);
    header["vertexBufferSize"] = json!(vertex_size);
    header["indexBufferOffset"] = json!(vertex_size);
    header["indexBufferSize"] = json!(index_size);
    edited["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["renderBuffer"]["Bytes"] =
        json!(STANDARD.encode(bytes));
    if let Some(appearances) = edited["Data"]["RootChunk"]["appearances"].as_array_mut() {
        for appearance in appearances {
            if let Some(materials) = appearance["Data"]["chunkMaterials"].as_array() {
                let originals = materials.clone();
                if !originals.is_empty() {
                    appearance["Data"]["chunkMaterials"] = json!(
                        chunks
                            .iter()
                            .map(|c| originals[c.original % originals.len()].clone())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    decode_geometry(&edited, false)?;
    report.chunks = chunks.len();
    *document = edited;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crossing_triangle_is_split_with_barycentric_attributes() {
        let p = [
            Point {
                position: Vec3::new(-2., 0., 0.),
                weights: [1., 0., 0.],
            },
            Point {
                position: Vec3::new(2., 0., 0.),
                weights: [0., 1., 0.],
            },
            Point {
                position: Vec3::new(0., 2., 0.),
                weights: [0., 0., 1.],
            },
        ];
        let polygon = clip(&p, [1., 0., 0., 0.], true);
        assert_eq!(polygon.len(), 4);
        assert!(
            polygon.iter().all(|v| v.position.x <= 0.000001
                && (v.weights.iter().sum::<f32>() - 1.).abs() < 0.00001)
        );
        assert!(
            polygon.iter().any(
                |v| (v.position - Vec3::ZERO).length() < 0.00001 && v.weights == [0.5, 0.5, 0.]
            )
        );
    }
}
