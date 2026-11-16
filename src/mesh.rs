//! Native Cyberpunk 2077 `.mesh` to binary glTF export.

use crate::{codec, schema::RedSchema};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use half::f16;
use serde_json::{Map, Value, json};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};
use thiserror::Error;

mod cut;
pub use cut::{CutReport, subtract_convex_volume};

const GLB_MAGIC: u32 = 0x4654_6C67;
const GLB_JSON_CHUNK: u32 = 0x4E4F_534A;
const GLB_BIN_CHUNK: u32 = 0x004E_4942;
const GLTF_FLOAT: u32 = 5126;
const GLTF_UNSIGNED_SHORT: u32 = 5123;

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("could not decode mesh CR2W: {0}")]
    Codec(#[from] codec::CodecError),
    #[error("could not access mesh output: {0}")]
    Io(#[from] io::Error),
    #[error("could not decode mesh buffer: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("malformed mesh: {0}")]
    Malformed(String),
}

/// Decoded native mesh geometry, independent of export file formats.
#[derive(Clone, Debug)]
pub struct MeshChunk {
    /// Original render-chunk index, preserved across LOD filtering.
    pub chunk_index: usize,
    /// Native chunk label.
    pub name: String,
    /// Decoded positions in exported glTF coordinates (metres, Y up).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals; empty when the source has no normal stream.
    pub normals: Vec<[f32; 3]>,
    /// Tangent directions and handedness.
    pub tangents: Vec<[f32; 4]>,
    /// Vertex colors.
    pub colors: Vec<[f32; 4]>,
    /// Primary texture coordinates.
    pub tex_coords: Vec<[f32; 2]>,
    /// Secondary texture coordinates.
    pub tex_coords_1: Vec<[f32; 2]>,
    /// Triangle indices.
    pub indices: Vec<u16>,
    /// Per-appearance material names in native appearance order.
    pub material_names: Vec<String>,
    /// Up to eight normalized bone influences per vertex; empty for rigid meshes.
    pub skin: Vec<VertexSkin>,
}

/// Native vertex influences into the mesh's named bone palette.
#[derive(Clone, Copy, Debug)]
pub struct VertexSkin {
    /// Bone palette indices; entries with zero weight are unused.
    pub joints: [u16; 8],
    /// Normalized linear-blend weights.
    pub weights: [f32; 8],
}

/// Mesh bone palette and authoritative inverse bind matrices in Y-up coordinates.
#[derive(Clone, Debug)]
pub struct MeshSkin {
    /// Palette names in native index order.
    pub names: Vec<String>,
    /// Model-to-bone rest matrices, matching `names`.
    pub inverse_bind: Vec<glam::Mat4>,
}

/// Authored inline garment deformation deltas for one native render chunk.
#[derive(Clone, Debug)]
pub struct GarmentMorph {
    /// Original render-chunk index, including native LOD/material splits.
    pub chunk_index: usize,
    /// Position deltas in metres, converted to the mesh decoder's Y-up basis.
    pub offsets: Vec<glam::Vec3>,
}

/// Reads inline half-float garment morphs from the native position/skin stream.
/// These can exist without a separate `garmentMeshParamGarment` proxy.
/// The fourth packed component and all source bytes remain unchanged.
///
/// # Errors
/// Returns [`MeshError`] for unsupported layouts, truncated or nonfinite data.
pub fn decode_garment_morphs(document: &Value) -> Result<Vec<GarmentMorph>, MeshError> {
    let root = member_path(document, &["Data", "RootChunk"])?;
    let custom = root["parameters"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| p.get("Data").unwrap_or(p))
        .any(|p| p["$type"] == "meshMeshParamGarmentSupport" && p["customMorph"] == true);
    if !custom {
        return Ok(Vec::new());
    }
    let blob = member_path(root, &["renderResourceBlob", "Data"])?;
    let header = member(blob, "header")?;
    let bytes = STANDARD.decode(string_member(member(blob, "renderBuffer")?, "Bytes")?)?;
    let mut result = Vec::new();
    for (chunk_index, info) in array_member(header, "renderChunkInfos")?.iter().enumerate() {
        let vertices = member(info, "chunkVertices")?;
        let layout = member(vertices, "vertexLayout")?;
        let elements = array_member(layout, "elements")?;
        let Some(extra) = elements.iter().find(|e| e["usage"] == "PS_ExtraData") else {
            continue;
        };
        if extra["type"] != "PT_Float16_4" || extra["streamIndex"].as_u64().unwrap_or(0) != 0 {
            return Err(malformed("unsupported inline garment morph packing"));
        }
        let influences = elements
            .iter()
            .filter(|e| e["usage"] == "PS_SkinIndices")
            .count()
            * 4;
        if !matches!(influences, 0 | 4 | 8) {
            return Err(malformed("unsupported garment skin influence count"));
        }
        let offsets = usize_array(member(vertices, "byteOffsets")?)?;
        let strides = usize_array(member(layout, "slotStrides")?)?;
        let stride = *strides
            .first()
            .ok_or_else(|| malformed("garment position stride absent"))?;
        let offset = *offsets
            .first()
            .ok_or_else(|| malformed("garment position offset absent"))?;
        let extra_offset = 8 + influences * 2;
        if stride < extra_offset + 8 {
            return Err(malformed("garment morph exceeds position stride"));
        }
        let count = recovered_vertex_count(
            usize_member(info, "numVertices")?,
            stride,
            &offsets,
            &strides,
        );
        let mut morphs = Vec::with_capacity(count);
        for index in 0..count {
            let at = checked_element_offset(offset, stride, index, extra_offset + 8, bytes.len())?
                + extra_offset;
            let read = |axis: usize| {
                f16::from_bits(u16::from_le_bytes([
                    bytes[at + axis * 2],
                    bytes[at + axis * 2 + 1],
                ]))
                .to_f32()
            };
            let delta = glam::Vec3::new(read(0), read(2), -read(1));
            if !delta.is_finite() {
                return Err(malformed("nonfinite inline garment morph"));
            }
            morphs.push(delta);
        }
        result.push(GarmentMorph {
            chunk_index,
            offsets: morphs,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod inline_garment_tests {
    use super::*;
    #[test]
    fn inline_garment_morph_uses_extended_skin_stride_and_y_up_coordinates() {
        let mut bytes = vec![0_u8; 32];
        for (axis, value) in [-0.02_f32, 0.03, 0.04, 0.].into_iter().enumerate() {
            bytes[24 + axis * 2..26 + axis * 2]
                .copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        let mut document = json!({"Data":{"RootChunk":{"parameters":[{"Data":{"$type":"meshMeshParamGarmentSupport","customMorph":true}}],"renderResourceBlob":{"Data":{
            "header":{"renderChunkInfos":[{"numVertices":1,"chunkVertices":{"byteOffsets":[0],"vertexLayout":{"slotStrides":[32],"elements":[{"usage":"PS_Position","type":"PT_Short4N"},{"usage":"PS_SkinIndices"},{"usage":"PS_SkinIndices","usageIndex":1},{"usage":"PS_ExtraData","type":"PT_Float16_4"}]}}}]},
            "renderBuffer":{"Bytes":STANDARD.encode(&bytes)}
        }}}}});
        let morphs = decode_garment_morphs(&document).unwrap();
        assert_eq!(morphs[0].chunk_index, 0);
        assert!(morphs[0].offsets[0].distance(glam::Vec3::new(-0.02, 0.04, -0.03)) < 0.0001);
        bytes.pop();
        document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["renderBuffer"]["Bytes"] =
            json!(STANDARD.encode(bytes));
        assert!(decode_garment_morphs(&document).is_err());
    }
}

/// Decodes a mesh's authored bone palette and inverse bind transforms.
///
/// # Errors
/// Returns [`MeshError`] for mismatched, singular or nonfinite bind matrices.
pub fn decode_skin(document: &Value) -> Result<Option<MeshSkin>, MeshError> {
    let root = member_path(document, &["Data", "RootChunk"])?;
    let Some(names) = root
        .get("boneNames")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
    else {
        return Ok(None);
    };
    let names = names
        .iter()
        .map(red_string)
        .collect::<Result<Vec<_>, _>>()?;
    let matrices = array_member(root, "boneRigMatrices")?;
    if matrices.len() != names.len() {
        return Err(malformed(
            "bone names and inverse bind matrices differ in length",
        ));
    }
    let basis = glam::Mat4::from_cols_array(&[
        1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
    ]);
    let inverse_bind = matrices
        .iter()
        .map(|matrix| {
            let mut columns = [[0.0_f32; 4]; 4];
            for (column, key) in ["X", "Y", "Z", "W"].iter().enumerate() {
                for (row, axis) in ["X", "Y", "Z", "W"].iter().enumerate() {
                    let value = member(member(matrix, key)?, axis)?
                        .as_f64()
                        .ok_or_else(|| malformed("invalid bind matrix component"))?;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "native matrix elements use f32"
                    )]
                    {
                        columns[column][row] = value as f32;
                    }
                }
            }
            let matrix = glam::Mat4::from_cols_array_2d(&columns);
            if !matrix.is_finite() || matrix.determinant().abs() < 1e-12 {
                return Err(malformed("nonfinite or singular inverse bind matrix"));
            }
            Ok(basis * matrix * basis.transpose())
        })
        .collect::<Result<Vec<_>, MeshError>>()?;
    Ok(Some(MeshSkin {
        names,
        inverse_bind,
    }))
}

/// Replaces a native mesh's inverse bind matrices using Y-up coordinates.
/// Palette order, skin weights and all vertex streams are retained.
///
/// # Errors
/// Returns an error for absent skinning, mismatched counts or invalid matrices.
/// The document is unchanged on failure.
pub fn replace_inverse_bind(
    document: &mut Value,
    matrices: &[glam::Mat4],
) -> Result<(), MeshError> {
    let skin =
        decode_skin(document)?.ok_or_else(|| malformed("replacement mesh has no skin palette"))?;
    if skin.names.len() != matrices.len()
        || matrices.iter().any(|m| {
            !m.is_finite()
                || m.determinant().abs() < 1e-12
                || !m.inverse().is_finite()
                || m.x_axis.w.abs() > 1e-5
                || m.y_axis.w.abs() > 1e-5
                || m.z_axis.w.abs() > 1e-5
                || (m.w_axis.w - 1.).abs() > 1e-5
        })
    {
        return Err(malformed("invalid replacement inverse bind matrices"));
    }
    let basis = glam::Mat4::from_cols_array(&[
        1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
    ]);
    for (value, matrix) in document["Data"]["RootChunk"]["boneRigMatrices"]
        .as_array_mut()
        .ok_or_else(|| malformed("mesh bind matrices absent"))?
        .iter_mut()
        .zip(matrices)
    {
        let columns = (basis.transpose() * *matrix * basis).to_cols_array_2d();
        for (column, key) in ["X", "Y", "Z", "W"].into_iter().enumerate() {
            for (row, axis) in ["X", "Y", "Z", "W"].into_iter().enumerate() {
                value[key][axis] = json!(columns[column][row]);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct GlbBuilder {
    binary: Vec<u8>,
    buffer_views: Vec<Value>,
    accessors: Vec<Value>,
}

impl GlbBuilder {
    fn push_f32_vectors<const N: usize>(
        &mut self,
        values: &[[f32; N]],
        accessor_type: &str,
        bounds: Option<(&[f32; N], &[f32; N])>,
    ) -> usize {
        self.align_binary();
        let byte_offset = self.binary.len();
        for value in values {
            for component in value {
                self.binary.extend_from_slice(&component.to_le_bytes());
            }
        }
        let view = self.push_view(byte_offset, self.binary.len() - byte_offset);
        let mut accessor = Map::from_iter([
            ("bufferView".to_owned(), json!(view)),
            ("componentType".to_owned(), json!(GLTF_FLOAT)),
            ("count".to_owned(), json!(values.len())),
            ("type".to_owned(), json!(accessor_type)),
        ]);
        if let Some((minimum, maximum)) = bounds {
            accessor.insert("min".to_owned(), json!(&minimum[..]));
            accessor.insert("max".to_owned(), json!(&maximum[..]));
        }
        self.accessors.push(Value::Object(accessor));
        self.accessors.len() - 1
    }

    fn push_indices(&mut self, values: &[u16]) -> usize {
        self.align_binary();
        let byte_offset = self.binary.len();
        for value in values {
            self.binary.extend_from_slice(&value.to_le_bytes());
        }
        let view = self.push_view(byte_offset, self.binary.len() - byte_offset);
        self.accessors.push(json!({
            "bufferView": view,
            "componentType": GLTF_UNSIGNED_SHORT,
            "count": values.len(),
            "type": "SCALAR"
        }));
        self.accessors.len() - 1
    }

    fn push_view(&mut self, byte_offset: usize, byte_length: usize) -> usize {
        self.buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length
        }));
        self.buffer_views.len() - 1
    }

    fn align_binary(&mut self) {
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
    }
}

/// Decodes render chunks from a reflected mesh document.
///
/// Positions, normals and winding use the same coordinate conversion as native
/// GLB export. `lod_filter` selects chunks whose native LOD mask is one.
///
/// # Errors
///
/// Returns [`MeshError`] for unsupported or malformed vertex/index buffers.
pub fn decode_geometry(document: &Value, lod_filter: bool) -> Result<Vec<MeshChunk>, MeshError> {
    decode_meshes(document, lod_filter)
}

/// Samples a native fluid vertex-animation frame into a collapsed render chunk.
///
/// Two interleaved RGBA texels store each point's normalized 16-bit XYZ position;
/// `PS_DestructionIndices.y` addresses the authored point shared by UV seams.
/// Positions use the same RED-to-preview coordinate conversion as mesh decoding.
///
/// # Errors
/// Returns [`MeshError`] for incompatible texture dimensions, missing addressing
/// streams, invalid point indices, frame indices or translation ranges.
pub fn apply_fluid_vertex_frame(
    document: &Value,
    chunk: &mut MeshChunk,
    texture: &image::RgbaImage,
    frame: u32,
    pieces: u32,
    translation: [f32; 2],
) -> Result<(), MeshError> {
    if pieces == 0
        || pieces.checked_mul(2) != Some(texture.width())
        || frame >= texture.height()
        || !translation.iter().all(|v| v.is_finite())
        || translation[1] <= translation[0]
    {
        return Err(malformed("invalid fluid vertex-animation texture or range"));
    }
    let blob = member_path(
        document,
        &["Data", "RootChunk", "renderResourceBlob", "Data"],
    )?;
    let native = member_path(blob, &["header", "renderChunkInfos"])?
        .as_array()
        .and_then(|chunks| chunks.get(chunk.chunk_index))
        .ok_or_else(|| malformed("fluid chunk absent"))?;
    let vertices = member(native, "chunkVertices")?;
    let offsets = usize_array(member(vertices, "byteOffsets")?)?;
    let layout = member(vertices, "vertexLayout")?;
    let elements = array_member(layout, "elements")?;
    let offset = offset_for_usage(elements, &offsets, "PS_DestructionIndices", 0)?
        .ok_or_else(|| malformed("fluid mesh has no native point addressing stream"))?;
    let element = elements
        .iter()
        .find(|v| v["usage"] == "PS_DestructionIndices")
        .ok_or_else(|| malformed("fluid point declaration absent"))?;
    if element["type"] != "PT_UShort2" {
        return Err(malformed("unsupported fluid point addressing format"));
    }
    let stream = element["streamIndex"].as_u64().unwrap_or(0) as usize;
    let strides = usize_array(member(layout, "slotStrides")?)?;
    let stride = *strides
        .get(stream)
        .ok_or_else(|| malformed("fluid address stride absent"))?;
    let bytes = STANDARD.decode(string_member(member(blob, "renderBuffer")?, "Bytes")?)?;
    for (index, position) in chunk.positions.iter_mut().enumerate() {
        let address = checked_element_offset(offset, stride, index, 4, bytes.len())?;
        let point = u32::from(read_u16(&bytes, address + 2)?);
        if point >= pieces {
            return Err(malformed(
                "fluid point index exceeds authored texture block",
            ));
        }
        let first = texture.get_pixel(point * 2, frame);
        let second = texture.get_pixel(point * 2 + 1, frame);
        let packed = [
            u16::from_le_bytes([first[0], first[1]]),
            u16::from_le_bytes([first[2], first[3]]),
            u16::from_le_bytes([second[0], second[1]]),
        ];
        let red: [f32; 3] = std::array::from_fn(|axis| {
            (f32::from(packed[axis]) / 65_535.)
                .mul_add(translation[1] - translation[0], translation[0])
        });
        *position = [red[0], red[2], -red[1]];
    }
    let mut normals = vec![glam::Vec3::ZERO; chunk.positions.len()];
    for triangle in chunk.indices.chunks_exact(3) {
        let a = glam::Vec3::from(chunk.positions[usize::from(triangle[0])]);
        let b = glam::Vec3::from(chunk.positions[usize::from(triangle[1])]);
        let c = glam::Vec3::from(chunk.positions[usize::from(triangle[2])]);
        let normal = (b - a).cross(c - a);
        for index in triangle {
            normals[usize::from(*index)] += normal;
        }
    }
    chunk.normals.clear();
    chunk.tangents.clear();
    for normal in normals {
        let normal = normal.normalize_or_zero();
        chunk.normals.push(normal.to_array());
        let tangent = normal
            .cross(if normal.z.abs() < 0.9 {
                glam::Vec3::Z
            } else {
                glam::Vec3::X
            })
            .normalize_or_zero();
        chunk.tangents.push([tangent.x, tangent.y, tangent.z, 1.]);
    }
    Ok(())
}

/// Replaces vertex positions, normals and tangents in a decoded native mesh.
///
/// All render chunks, including lower LODs, must be supplied. The original index,
/// UV, colour and skin streams are retained. Positions are requantized together
/// so edited shapes do not silently clip at the original quantization limits.
///
/// # Errors
/// Returns an error for incomplete chunk coverage, invalid vertices or layouts.
pub fn replace_geometry(document: &mut Value, chunks: &[MeshChunk]) -> Result<(), MeshError> {
    let root = member_path(document, &["Data", "RootChunk"])?;
    let blob = member_path(root, &["renderResourceBlob", "Data"])?;
    let header = member(blob, "header")?;
    let native_chunks = array_member(header, "renderChunkInfos")?;
    if chunks.len() != native_chunks.len() {
        return Err(malformed("replacement must include every native LOD chunk"));
    }
    let mut bytes = STANDARD.decode(string_member(member(blob, "renderBuffer")?, "Bytes")?)?;
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for chunk in chunks {
        for point in &chunk.positions {
            let red = [point[0], -point[2], point[1]];
            for axis in 0..3 {
                if !red[axis].is_finite() {
                    return Err(malformed("replacement positions must be finite"));
                }
                minimum[axis] = minimum[axis].min(red[axis]);
                maximum[axis] = maximum[axis].max(red[axis]);
            }
        }
    }
    let translation: [f32; 3] = std::array::from_fn(|i| (minimum[i] + maximum[i]) * 0.5);
    let scale: [f32; 3] = std::array::from_fn(|i| ((maximum[i] - minimum[i]) * 0.5).max(0.000001));
    if translation.iter().chain(&scale).any(|v| !v.is_finite()) {
        return Err(malformed("replacement geometry has invalid bounds"));
    }
    let mut seen = std::collections::HashSet::new();
    for chunk in chunks {
        if !seen.insert(chunk.chunk_index) {
            return Err(malformed("duplicate replacement chunk"));
        }
        let native = native_chunks
            .get(chunk.chunk_index)
            .ok_or_else(|| malformed("replacement chunk index exceeds native layout"))?;
        let count = usize_member(native, "numVertices")?;
        if count != chunk.positions.len()
            || (!chunk.normals.is_empty() && chunk.normals.len() != count)
            || (!chunk.tangents.is_empty() && chunk.tangents.len() != count)
        {
            return Err(malformed(
                "replacement vertex count differs from native layout",
            ));
        }
        let vertices = member(native, "chunkVertices")?;
        let layout = member(vertices, "vertexLayout")?;
        let offsets = usize_array(member(vertices, "byteOffsets")?)?;
        let strides = usize_array(member(layout, "slotStrides")?)?;
        let elements = array_member(layout, "elements")?;
        let position = offset_for_usage(elements, &offsets, "PS_Position", 0)?
            .ok_or_else(|| malformed("missing native position stream"))?;
        let stride = *strides
            .first()
            .ok_or_else(|| malformed("missing native position stride"))?;
        let normal = offset_for_usage(elements, &offsets, "PS_Normal", 0)?;
        let tangent = offset_for_usage(elements, &offsets, "PS_Tangent", 0)?;
        let normal_stride = if tangent.is_some() { 8 } else { 4 };
        for index in 0..count {
            let start = checked_element_offset(position, stride, index, 6, bytes.len())?;
            let point = chunk.positions[index];
            let red = [point[0], -point[2], point[1]];
            for axis in 0..3 {
                let quantized = (((red[axis] - translation[axis]) / scale[axis]).clamp(-1., 1.)
                    * 32767.)
                    .round() as i16;
                bytes[start + axis * 2..start + axis * 2 + 2]
                    .copy_from_slice(&quantized.to_le_bytes());
            }
            if let Some(offset) = normal
                && !chunk.normals.is_empty()
            {
                let start = checked_element_offset(offset, normal_stride, index, 4, bytes.len())?;
                bytes[start..start + 4]
                    .copy_from_slice(&encode_direction(chunk.normals[index], 1.)?.to_le_bytes());
            }
            if let Some(offset) = tangent
                && !chunk.tangents.is_empty()
            {
                let start = checked_element_offset(
                    offset + usize::from(normal.is_some()) * 4,
                    normal_stride,
                    index,
                    4,
                    bytes.len(),
                )?;
                let value = chunk.tangents[index];
                bytes[start..start + 4].copy_from_slice(
                    &encode_direction([value[0], value[1], value[2]], value[3])?.to_le_bytes(),
                );
            }
        }
    }
    let header = &mut document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["header"];
    for (field, values) in [
        ("quantizationOffset", translation),
        ("quantizationScale", scale),
    ] {
        for (axis, name) in ["X", "Y", "Z"].iter().enumerate() {
            header[field][*name] = serde_json::json!(values[axis]);
        }
    }
    document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["renderBuffer"]["Bytes"] =
        Value::String(STANDARD.encode(bytes));
    let bounds = &mut document["Data"]["RootChunk"]["boundingBox"];
    for (field, values) in [("Min", minimum), ("Max", maximum)] {
        for (axis, name) in ["X", "Y", "Z"].iter().enumerate() {
            bounds[field][*name] = serde_json::json!(values[axis]);
        }
    }
    Ok(())
}

fn encode_direction(value: [f32; 3], handedness: f32) -> Result<u32, MeshError> {
    if value.iter().any(|v| !v.is_finite()) || !handedness.is_finite() {
        return Err(malformed("replacement directions must be finite"));
    }
    let value = normalize(value);
    let red = [value[0], -value[2], value[1]];
    let mut packed = if handedness < 0. { 3_u32 << 30 } else { 0 };
    for (axis, value) in red.into_iter().enumerate() {
        packed |= (((value.clamp(-1., 1.) + 1.) * 511.5).round() as u32) << (axis * 10);
    }
    Ok(packed)
}

/// Exports the first-level-of-detail geometry in a Cyberpunk `.mesh` as GLB.
///
/// The GLB includes WolvenKit-compatible `materialNames` extras. A neighboring
/// `.Material.json` and its material repository can therefore be consumed by
/// the `WolvenKit` Blender add-on without changing the geometry importer.
///
/// # Errors
///
/// Returns [`MeshError`] when CR2W decoding fails, the mesh buffer is malformed,
/// or the output cannot be written.
pub fn export_glb(
    input: &Path,
    schema: &RedSchema,
    output: &Path,
    lod_filter: bool,
) -> Result<(), MeshError> {
    let document = codec::decode_wkit_with_red_schema(input, schema)?;
    let meshes = decode_meshes(&document, lod_filter)?;
    let glb = build_glb(&meshes)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, glb)?;
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "keeping the vertex stream layout together makes binary-format auditing safer"
)]
fn decode_meshes(document: &Value, lod_filter: bool) -> Result<Vec<MeshChunk>, MeshError> {
    let root = member_path(document, &["Data", "RootChunk"])?;
    let render_blob = member_path(root, &["renderResourceBlob", "Data"])?;
    let header = member(render_blob, "header")?;
    let render_buffer = string_member(member(render_blob, "renderBuffer")?, "Bytes")?;
    let render_buffer = STANDARD.decode(render_buffer)?;
    let quantization_scale = quantization_vector(member(header, "quantizationScale")?)?;
    let quantization_offset = quantization_vector(member(header, "quantizationOffset")?)?;
    let index_buffer_offset = usize_member(header, "indexBufferOffset")?;
    let appearances = decode_appearances(root)?;
    let chunks = array_member(header, "renderChunkInfos")?;
    let mut meshes = Vec::with_capacity(chunks.len());

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let lod = u64_member_or_default(chunk, "lodMask")?;
        if lod_filter && lod != 1 {
            continue;
        }
        let declared_vertex_count = usize_member(chunk, "numVertices")?;
        let index_count = usize_member(chunk, "numIndices")?;
        let vertices = member(chunk, "chunkVertices")?;
        let layout = member(vertices, "vertexLayout")?;
        let byte_offsets = usize_array(member(vertices, "byteOffsets")?)?;
        let slot_strides = usize_array(member(layout, "slotStrides")?)?;
        let elements = array_member(layout, "elements")?;
        let position_offset = offset_for_usage(elements, &byte_offsets, "PS_Position", 0)?
            .ok_or_else(|| malformed("vertex layout has no position stream"))?;
        let position_stride = *slot_strides
            .first()
            .ok_or_else(|| malformed("vertex layout has no position stride"))?;
        // Several shipped terrain chunks have a wrapped 16-bit numVertices.
        // Their packed position and UV spans independently retain the full
        // count. Only recover an exact 65536 wrap confirmed by both streams.
        let vertex_count = recovered_vertex_count(
            declared_vertex_count,
            position_stride,
            &byte_offsets,
            &slot_strides,
        );
        let tex_coord_offset = offset_for_usage(elements, &byte_offsets, "PS_TexCoord", 0)?;
        let tex_coord_1_offset = offset_for_usage(elements, &byte_offsets, "PS_TexCoord", 1)?;
        let normal_offset = offset_for_usage(elements, &byte_offsets, "PS_Normal", 0)?;
        let tangent_offset = offset_for_usage(elements, &byte_offsets, "PS_Tangent", 0)?;
        let color_offset = offset_for_usage(elements, &byte_offsets, "PS_Color", 0)?;
        let normal_stride = if tangent_offset.is_some() { 8 } else { 4 };
        let color_stride = if tex_coord_1_offset.is_some() { 8 } else { 4 };

        let positions = decode_positions(
            &render_buffer,
            position_offset,
            position_stride,
            vertex_count,
            quantization_scale,
            quantization_offset,
        )?;
        let skin = decode_vertex_skin(
            &render_buffer,
            position_offset,
            position_stride,
            vertex_count,
            elements,
        )?;
        let normals = normal_offset.map_or_else(
            || Ok(Vec::new()),
            |offset| decode_normals(&render_buffer, offset, normal_stride, vertex_count),
        )?;
        let tangents = tangent_offset.map_or_else(
            || Ok(Vec::new()),
            |offset| {
                decode_tangents(
                    &render_buffer,
                    offset + usize::from(normal_offset.is_some()) * 4,
                    normal_stride,
                    vertex_count,
                )
            },
        )?;
        let colors = color_offset.map_or_else(
            || Ok(Vec::new()),
            |offset| decode_colors(&render_buffer, offset, color_stride, vertex_count),
        )?;
        let tex_coords = tex_coord_offset.map_or_else(
            || Ok(Vec::new()),
            |offset| decode_tex_coords(&render_buffer, offset, 4, vertex_count, true),
        )?;
        let tex_coords_1 = tex_coord_1_offset.map_or_else(
            || Ok(Vec::new()),
            |offset| {
                decode_tex_coords(
                    &render_buffer,
                    offset + usize::from(color_offset.is_some()) * 4,
                    color_stride,
                    vertex_count,
                    false,
                )
            },
        )?;
        let relative_indices_offset = member(chunk, "chunkIndices")?
            .get("teOffset")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| malformed("index buffer offset exceeds platform size"))?
            .unwrap_or(0);
        let indices_offset = index_buffer_offset
            .checked_add(relative_indices_offset)
            .ok_or_else(|| malformed("index buffer offset overflow"))?;
        let indices = decode_indices(&render_buffer, indices_offset, index_count, positions.len())?;
        let material_names = appearances
            .iter()
            .map(|materials| material_for_chunk(materials, chunk_index, chunks.len()))
            .collect();

        meshes.push(MeshChunk {
            chunk_index,
            name: format!("submesh_{chunk_index:02}_LOD_{lod}"),
            positions,
            normals,
            tangents,
            colors,
            tex_coords,
            tex_coords_1,
            indices,
            material_names,
            skin,
        });
    }
    if meshes.is_empty() {
        return Err(malformed("mesh has no exportable render chunks"));
    }
    Ok(meshes)
}

fn recovered_vertex_count(
    declared: usize,
    position_stride: usize,
    offsets: &[usize],
    strides: &[usize],
) -> usize {
    if position_stride == 0 || offsets.len() < 3 || strides.get(1) != Some(&4) {
        return declared;
    }
    let Some(position_bytes) = offsets[1].checked_sub(offsets[0]) else {
        return declared;
    };
    let Some(uv_bytes) = offsets[2].checked_sub(offsets[1]) else {
        return declared;
    };
    let stream_count = position_bytes / position_stride;
    let count = declared + (stream_count.saturating_sub(declared) / 65_536) * 65_536;
    // Stream starts are 16-byte aligned, so allow only alignment padding.
    if count > declared
        && position_bytes >= count * position_stride
        && position_bytes - count * position_stride < 16
        && uv_bytes >= count * 4
        && uv_bytes - count * 4 < 16
    {
        count
    } else {
        declared
    }
}

/// Replaces only native skin indices and normalized byte weights. The palette,
/// geometry, UVs, materials and other vertex streams remain unchanged.
///
/// # Errors
/// Returns errors for invalid layouts, palette indices or skin stream lengths.
pub fn replace_skin_weights(document: &mut Value, chunks: &[MeshChunk]) -> Result<(), MeshError> {
    let palette_len = decode_skin(document)?.map_or(0, |skin| skin.names.len());
    let blob = member_path(
        document,
        &["Data", "RootChunk", "renderResourceBlob", "Data"],
    )?;
    let header = member(blob, "header")?;
    let native = array_member(header, "renderChunkInfos")?;
    let mut bytes = STANDARD.decode(string_member(member(blob, "renderBuffer")?, "Bytes")?)?;
    for chunk in chunks.iter().filter(|c| !c.skin.is_empty()) {
        let info = native
            .get(chunk.chunk_index)
            .ok_or_else(|| malformed("skin replacement chunk absent"))?;
        let vertices = member(info, "chunkVertices")?;
        let layout = member(vertices, "vertexLayout")?;
        let elements = array_member(layout, "elements")?;
        let influences = elements
            .iter()
            .filter(|e| e["usage"] == "PS_SkinIndices")
            .count()
            * 4;
        if !matches!(influences, 4 | 8) || chunk.skin.len() != chunk.positions.len() {
            return Err(malformed("invalid skin replacement layout"));
        }
        let offsets = usize_array(member(vertices, "byteOffsets")?)?;
        let strides = usize_array(member(layout, "slotStrides")?)?;
        let offset = offset_for_usage(elements, &offsets, "PS_Position", 0)?
            .ok_or_else(|| malformed("skin position stream absent"))?;
        let stride = *strides
            .first()
            .ok_or_else(|| malformed("skin stride absent"))?;
        let count = recovered_vertex_count(
            usize_member(info, "numVertices")?,
            stride,
            &offsets,
            &strides,
        );
        if chunk.skin.len() != count {
            return Err(malformed(
                "skin replacement differs from native vertex count",
            ));
        }
        if stride < 8 + influences * 2 {
            return Err(malformed("skin replacement exceeds stride"));
        }
        for (index, skin) in chunk.skin.iter().enumerate() {
            if skin.weights.iter().any(|w| !w.is_finite() || *w < 0.) {
                return Err(malformed("nonfinite or negative skin replacement weight"));
            }
            let mut pairs: Vec<_> = skin
                .joints
                .iter()
                .copied()
                .zip(skin.weights)
                .filter(|(_, w)| *w > 0.)
                .collect();
            if pairs
                .iter()
                .any(|(j, _)| *j > 255 || usize::from(*j) >= palette_len)
            {
                return Err(malformed("invalid skin replacement influence"));
            }
            if pairs.len() > influences {
                return Err(malformed(
                    "skin replacement exceeds native influence capacity",
                ));
            }
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let sum: f32 = pairs.iter().map(|(_, w)| w).sum();
            if sum <= 0. || !sum.is_finite() {
                return Err(malformed("zero skin replacement weight"));
            }
            let start =
                checked_element_offset(offset, stride, index, 8 + influences * 2, bytes.len())? + 8;
            bytes[start..start + influences * 2].fill(0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "validated positive normalized weights quantize to bytes"
            )]
            let mut quantized: Vec<_> = pairs
                .iter()
                .map(|(_, w)| (*w / sum * 255.).floor() as u8)
                .collect();
            let remaining = 255usize - quantized.iter().map(|w| usize::from(*w)).sum::<usize>();
            for i in 0..remaining {
                quantized[i % pairs.len()] += 1;
            }
            for (i, ((joint, _), weight)) in pairs.iter().zip(quantized).enumerate() {
                bytes[start + i] = u8::try_from(*joint)
                    .map_err(|_| malformed("skin palette index exceeds byte range"))?;
                bytes[start + influences + i] = weight;
            }
        }
    }
    document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["renderBuffer"]["Bytes"] =
        Value::String(STANDARD.encode(bytes));
    Ok(())
}

fn decode_vertex_skin(
    buffer: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
    elements: &[Value],
) -> Result<Vec<VertexSkin>, MeshError> {
    let sets = elements
        .iter()
        .filter(|e| e.get("usage").and_then(Value::as_str) == Some("PS_SkinIndices"))
        .count();
    if sets == 0 {
        return Ok(Vec::new());
    }
    if sets > 2 {
        return Err(malformed("more than eight skin influences per vertex"));
    }
    let influences = sets * 4;
    if stride < 8 + influences * 2 {
        return Err(malformed("skin data exceeds position stream stride"));
    }
    (0..count)
        .map(|index| {
            let start = index
                .checked_mul(stride)
                .and_then(|v| v.checked_add(offset))
                .and_then(|v| v.checked_add(8))
                .ok_or_else(|| malformed("skin offset overflow"))?;
            let bytes = buffer
                .get(
                    start
                        ..start
                            .checked_add(influences * 2)
                            .ok_or_else(|| malformed("skin size overflow"))?,
                )
                .ok_or_else(|| malformed("truncated skin stream"))?;
            let mut result = VertexSkin {
                joints: [0; 8],
                weights: [0.0; 8],
            };
            for i in 0..influences {
                result.joints[i] = u16::from(bytes[i]);
                result.weights[i] = f32::from(bytes[influences + i]);
            }
            let total: f32 = result.weights.iter().sum();
            if total == 0.0 {
                // Native zero-weight vertices bind to palette bone zero.
                result.joints[0] = 0;
                result.weights[0] = 1.0;
            } else {
                result
                    .weights
                    .iter_mut()
                    .for_each(|weight| *weight /= total);
            }
            Ok(result)
        })
        .collect()
}

/// Replaces fixed-length native index streams, retaining every vertex attribute.
/// Degenerate triangles can mask covered regions without changing chunk slots.
///
/// # Errors
/// Returns an error for invalid chunk identities, counts or vertex indices.
pub fn replace_indices(document: &mut Value, chunks: &[MeshChunk]) -> Result<(), MeshError> {
    let blob = member_path(
        document,
        &["Data", "RootChunk", "renderResourceBlob", "Data"],
    )?;
    let header = member(blob, "header")?;
    let infos = array_member(header, "renderChunkInfos")?;
    let base = usize_member(header, "indexBufferOffset")?;
    let mut bytes = STANDARD.decode(string_member(member(blob, "renderBuffer")?, "Bytes")?)?;
    let mut seen = std::collections::HashSet::new();
    for chunk in chunks {
        if !seen.insert(chunk.chunk_index) {
            return Err(malformed("duplicate index replacement chunk"));
        }
        let info = infos
            .get(chunk.chunk_index)
            .ok_or_else(|| malformed("index replacement chunk absent"))?;
        if usize_member(info, "numIndices")? != chunk.indices.len()
            || !chunk.indices.len().is_multiple_of(3)
            || chunk
                .indices
                .iter()
                .any(|i| usize::from(*i) >= chunk.positions.len())
        {
            return Err(malformed("invalid replacement index stream"));
        }
        let relative = member(info, "chunkIndices")?
            .get("teOffset")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let offset = base
            .checked_add(
                usize::try_from(relative)
                    .map_err(|_| malformed("index offset exceeds platform size"))?,
            )
            .ok_or_else(|| malformed("index offset overflow"))?;
        for i in 0..chunk.indices.len() {
            let converted = match i % 3 {
                0 => i + 1,
                1 => i - 1,
                _ => i,
            };
            let index = chunk.indices[converted];
            let start = checked_element_offset(offset, 2, i, 2, bytes.len())?;
            bytes[start..start + 2].copy_from_slice(&index.to_le_bytes());
        }
    }
    document["Data"]["RootChunk"]["renderResourceBlob"]["Data"]["renderBuffer"]["Bytes"] =
        Value::String(STANDARD.encode(bytes));
    Ok(())
}

fn decode_appearances(root: &Value) -> Result<Vec<Vec<String>>, MeshError> {
    array_member(root, "appearances")?
        .iter()
        .map(|appearance| {
            let data = member(appearance, "Data")?;
            if let Some(materials) = data.get("chunkMaterials").and_then(Value::as_array) {
                materials.iter().map(red_string).collect()
            } else if root
                .get("materialEntries")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
            {
                Ok(Vec::new())
            } else {
                Err(malformed("mesh appearance has no material slots"))
            }
        })
        .collect()
}

fn material_for_chunk(materials: &[String], chunk_index: usize, chunk_count: usize) -> String {
    if materials.is_empty() {
        return "default".to_owned();
    }
    if chunk_index < materials.len() {
        return materials[chunk_index].clone();
    }
    let repeated_index = (chunk_index - materials.len()) % materials.len().min(chunk_count);
    materials[repeated_index].clone()
}

fn decode_positions(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
    scale: [f32; 3],
    translation: [f32; 3],
) -> Result<Vec<[f32; 3]>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, stride, index, 6, bytes.len())?;
        let x = f32::from(read_i16(bytes, start)?) / 32_767.0 * scale[0] + translation[0];
        let y = f32::from(read_i16(bytes, start + 2)?) / 32_767.0 * scale[1] + translation[1];
        let z = f32::from(read_i16(bytes, start + 4)?) / 32_767.0 * scale[2] + translation[2];
        result.push([x, z, -y]);
    }
    Ok(result)
}

fn decode_normals(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
) -> Result<Vec<[f32; 3]>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, stride, index, 4, bytes.len())?;
        let packed = read_u32(bytes, start)?;
        let decoded = decode_dec4(packed);
        result.push(normalize([decoded[0], decoded[2], -decoded[1]]));
    }
    Ok(result)
}

fn decode_tangents(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
) -> Result<Vec<[f32; 4]>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, stride, index, 4, bytes.len())?;
        let decoded = decode_dec4(read_u32(bytes, start)?);
        let normalized = normalize([decoded[0], decoded[2], -decoded[1]]);
        result.push([normalized[0], normalized[1], normalized[2], decoded[3]]);
    }
    Ok(result)
}

fn decode_colors(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
) -> Result<Vec<[f32; 4]>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, stride, index, 4, bytes.len())?;
        result.push([
            f32::from(bytes[start]) / 255.0,
            f32::from(bytes[start + 1]) / 255.0,
            f32::from(bytes[start + 2]) / 255.0,
            f32::from(bytes[start + 3]) / 255.0,
        ]);
    }
    Ok(result)
}

fn decode_tex_coords(
    bytes: &[u8],
    offset: usize,
    stride: usize,
    count: usize,
    flip_vertical: bool,
) -> Result<Vec<[f32; 2]>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, stride, index, 4, bytes.len())?;
        let vertical = f32::from(f16::from_bits(read_u16(bytes, start + 2)?));
        result.push([
            f32::from(f16::from_bits(read_u16(bytes, start)?)),
            if flip_vertical {
                1.0 - vertical
            } else {
                vertical
            },
        ]);
    }
    Ok(result)
}

fn decode_indices(
    bytes: &[u8],
    offset: usize,
    count: usize,
    vertex_count: usize,
) -> Result<Vec<u16>, MeshError> {
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let start = checked_element_offset(offset, 2, index, 2, bytes.len())?;
        let value = read_u16(bytes, start)?;
        if usize::from(value) >= vertex_count {
            return Err(malformed(format!(
                "index {value} exceeds vertex count {vertex_count}"
            )));
        }
        result.push(value);
    }
    for triangle in result.chunks_exact_mut(3) {
        triangle.swap(0, 1);
    }
    Ok(result)
}

fn build_glb(meshes: &[MeshChunk]) -> Result<Vec<u8>, MeshError> {
    let mut builder = GlbBuilder::default();
    let mut gltf_meshes = Vec::with_capacity(meshes.len());
    let mut nodes = Vec::with_capacity(meshes.len());

    for (mesh_index, mesh) in meshes.iter().enumerate() {
        let (minimum, maximum) = position_bounds(&mesh.positions)?;
        let positions =
            builder.push_f32_vectors(&mesh.positions, "VEC3", Some((&minimum, &maximum)));
        let mut attributes = Map::from_iter([("POSITION".to_owned(), json!(positions))]);
        if !mesh.normals.is_empty() {
            attributes.insert(
                "NORMAL".to_owned(),
                json!(builder.push_f32_vectors(&mesh.normals, "VEC3", None)),
            );
        }
        if !mesh.tangents.is_empty() {
            attributes.insert(
                "TANGENT".to_owned(),
                json!(builder.push_f32_vectors(&mesh.tangents, "VEC4", None)),
            );
        }
        if !mesh.colors.is_empty() {
            attributes.insert(
                "COLOR_0".to_owned(),
                json!(builder.push_f32_vectors(&mesh.colors, "VEC4", None)),
            );
        }
        if !mesh.tex_coords.is_empty() {
            attributes.insert(
                "TEXCOORD_0".to_owned(),
                json!(builder.push_f32_vectors(&mesh.tex_coords, "VEC2", None)),
            );
        }
        if !mesh.tex_coords_1.is_empty() {
            attributes.insert(
                "TEXCOORD_1".to_owned(),
                json!(builder.push_f32_vectors(&mesh.tex_coords_1, "VEC2", None)),
            );
        }
        let indices = builder.push_indices(&mesh.indices);
        gltf_meshes.push(json!({
            "name": mesh.name,
            "extras": {"materialNames": mesh.material_names},
            "primitives": [{
                "attributes": attributes,
                "indices": indices,
                "material": 0
            }]
        }));
        nodes.push(json!({"name": mesh.name, "mesh": mesh_index}));
    }

    builder.align_binary();
    let document = json!({
        "asset": {
            "generator": format!("ghostline-red {}", env!("CARGO_PKG_VERSION")),
            "version": "2.0"
        },
        "extras": {"ExperimentalMergedMeshes": false},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": (0..nodes.len()).collect::<Vec<_>>()}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": [{"name": "Default", "doubleSided": true, "pbrMetallicRoughness": {}}],
        "buffers": [{"byteLength": builder.binary.len()}],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors
    });
    encode_glb(&document, &builder.binary)
}

fn encode_glb(document: &Value, binary: &[u8]) -> Result<Vec<u8>, MeshError> {
    let mut json_bytes =
        serde_json::to_vec(document).map_err(|error| malformed(error.to_string()))?;
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let mut binary = binary.to_vec();
    while !binary.len().is_multiple_of(4) {
        binary.push(0);
    }
    let total_length = 12_usize
        .checked_add(8 + json_bytes.len())
        .and_then(|value| value.checked_add(8 + binary.len()))
        .ok_or_else(|| malformed("GLB length overflow"))?;
    let total_length =
        u32::try_from(total_length).map_err(|_| malformed("GLB exceeds four gigabytes"))?;
    let mut output = Vec::with_capacity(total_length as usize);
    output.write_all(&GLB_MAGIC.to_le_bytes())?;
    output.write_all(&2_u32.to_le_bytes())?;
    output.write_all(&total_length.to_le_bytes())?;
    output.write_all(
        &u32::try_from(json_bytes.len())
            .map_err(|_| malformed("GLB JSON chunk is too large"))?
            .to_le_bytes(),
    )?;
    output.write_all(&GLB_JSON_CHUNK.to_le_bytes())?;
    output.write_all(&json_bytes)?;
    output.write_all(
        &u32::try_from(binary.len())
            .map_err(|_| malformed("GLB binary chunk is too large"))?
            .to_le_bytes(),
    )?;
    output.write_all(&GLB_BIN_CHUNK.to_le_bytes())?;
    output.write_all(&binary)?;
    Ok(output)
}

fn position_bounds(positions: &[[f32; 3]]) -> Result<([f32; 3], [f32; 3]), MeshError> {
    let first = *positions
        .first()
        .ok_or_else(|| malformed("render chunk has no positions"))?;
    let mut minimum = first;
    let mut maximum = first;
    for position in &positions[1..] {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(position[axis]);
            maximum[axis] = maximum[axis].max(position[axis]);
        }
    }
    Ok((minimum, maximum))
}

fn offset_for_usage(
    elements: &[Value],
    byte_offsets: &[usize],
    usage: &str,
    usage_index: u64,
) -> Result<Option<usize>, MeshError> {
    let element = elements.iter().find(|element| {
        element.get("usage").and_then(Value::as_str) == Some(usage)
            && element
                .get("usageIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                == usage_index
    });
    let Some(element) = element else {
        return Ok(None);
    };
    let stream_index = element
        .get("streamIndex")
        .and_then(Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|_| malformed("vertex stream index exceeds platform size"))?
        .unwrap_or(0);
    byte_offsets
        .get(stream_index)
        .copied()
        .map(Some)
        .ok_or_else(|| malformed(format!("stream index {stream_index} is out of range")))
}

fn member_path<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, MeshError> {
    path.iter()
        .try_fold(value, |current, name| member(current, name))
}

fn member<'a>(value: &'a Value, name: &str) -> Result<&'a Value, MeshError> {
    value
        .get(name)
        .ok_or_else(|| malformed(format!("missing {name}")))
}

fn array_member<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], MeshError> {
    member(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| malformed(format!("{name} is not an array")))
}

fn string_member<'a>(value: &'a Value, name: &str) -> Result<&'a str, MeshError> {
    member(value, name)?
        .as_str()
        .ok_or_else(|| malformed(format!("{name} is not a string")))
}

fn usize_member(value: &Value, name: &str) -> Result<usize, MeshError> {
    usize::try_from(u64_member(value, name)?)
        .map_err(|_| malformed(format!("{name} exceeds platform size")))
}

fn u64_member(value: &Value, name: &str) -> Result<u64, MeshError> {
    member(value, name)?
        .as_u64()
        .ok_or_else(|| malformed(format!("{name} is not an unsigned integer")))
}

fn u64_member_or_default(value: &Value, name: &str) -> Result<u64, MeshError> {
    value.get(name).map_or(Ok(0), |member| {
        member
            .as_u64()
            .ok_or_else(|| malformed(format!("{name} is not an unsigned integer")))
    })
}

fn usize_array(value: &Value) -> Result<Vec<usize>, MeshError> {
    value
        .as_array()
        .ok_or_else(|| malformed("expected integer array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| malformed("array contains an invalid offset"))
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED Vector4 components are stored as 32-bit floats and JSON exposes them as f64"
)]
fn quantization_vector(value: &Value) -> Result<[f32; 3], MeshError> {
    let component = |name| {
        member(value, name)?
            .as_f64()
            .map(|number| number as f32)
            .ok_or_else(|| malformed(format!("{name} is not a number")))
    };
    Ok([component("X")?, component("Y")?, component("Z")?])
}

fn red_string(value: &Value) -> Result<String, MeshError> {
    value
        .get("$value")
        .or_else(|| value.get("DepotPath"))
        .or_else(|| value.as_str().map(|_| value))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| malformed("invalid RED string"))
}

fn checked_element_offset(
    base: usize,
    stride: usize,
    index: usize,
    width: usize,
    buffer_len: usize,
) -> Result<usize, MeshError> {
    let start = index
        .checked_mul(stride)
        .and_then(|offset| base.checked_add(offset))
        .ok_or_else(|| malformed("mesh buffer offset overflow"))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| malformed("mesh buffer range overflow"))?;
    if end > buffer_len {
        return Err(malformed(format!(
            "mesh buffer range {start}..{end} exceeds {buffer_len} bytes"
        )));
    }
    Ok(start)
}

fn read_i16(bytes: &[u8], offset: usize) -> Result<i16, MeshError> {
    Ok(i16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, MeshError> {
    Ok(u16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, MeshError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], MeshError> {
    bytes
        .get(offset..offset.saturating_add(N))
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| malformed(format!("could not read {N} bytes at offset {offset}")))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "10-bit packed components are exactly representable by f32"
)]
fn decode_dec4(value: u32) -> [f32; 4] {
    let scale = 1.0 / 1023.0;
    let component = |shift| (((value >> shift) & 0x3ff_u32) as f32 * 2.0 * scale) - 1.0;
    let w = match value >> 30 {
        0 => 1.0,
        3 => -1.0,
        _ => 0.0,
    };
    [component(0), component(10), component(20), w]
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length <= f32::EPSILON {
        return [0.0, 0.0, 1.0];
    }
    [value[0] / length, value[1] / length, value[2] / length]
}

fn malformed(reason: impl Into<String>) -> MeshError {
    MeshError::Malformed(reason.into())
}

#[cfg(test)]
mod tests {
    use super::{
        decode_dec4, encode_glb, material_for_chunk, normalize, quantization_vector,
        u64_member_or_default,
    };
    use serde_json::json;

    #[test]
    fn material_for_chunk_repeats_short_appearance_lists() {
        let materials = vec!["first".to_owned(), "second".to_owned()];

        assert_eq!(material_for_chunk(&materials, 3, 4), "second");
    }

    #[test]
    fn decode_dec4_maps_endpoints_to_signed_unit_range() {
        let decoded = decode_dec4(0x3ff | (0x3ff << 10) | (0x3ff << 20));

        assert!(
            decoded
                .iter()
                .all(|component| (*component - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn normalize_returns_a_unit_vector() {
        let normalized = normalize([3.0, 0.0, 4.0]);

        assert!((normalized[0] - 0.6).abs() < f32::EPSILON);
        assert!(normalized[1].abs() < f32::EPSILON);
        assert!((normalized[2] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn encode_glb_writes_glb_version_two_header() {
        let result = encode_glb(&json!({"asset": {"version": "2.0"}}), &[]).unwrap();

        assert_eq!(&result[..8], b"glTF\x02\0\0\0");
    }

    #[test]
    fn u64_member_or_default_returns_zero_for_omitted_red_property() {
        let value = json!({"$type": "rendChunk"});

        assert_eq!(u64_member_or_default(&value, "lodMask").unwrap(), 0);
    }

    #[test]
    fn quantization_vector_ignores_unused_non_numeric_w_component() {
        let value = json!({"$type": "Vector4", "X": 0.5, "Y": 1.0, "Z": 1.5, "W": null});
        let decoded = quantization_vector(&value).unwrap();

        assert!(
            decoded
                .iter()
                .zip([0.5, 1.0, 1.5])
                .all(|(actual, expected)| (*actual - expected).abs() < f32::EPSILON)
        );
    }
}
#[cfg(test)]
mod geometry_write_tests {
    #[test]
    fn replacement_requantizes_all_lods_and_preserves_indices() {
        use base64::Engine as _;
        let mut bytes = vec![0_u8; 108];
        for start in [48, 56, 64, 72, 80, 88] {
            bytes[start..start + 4].copy_from_slice(
                &super::encode_direction([0., 1., 0.], 1.)
                    .unwrap()
                    .to_le_bytes(),
            );
            bytes[start + 4..start + 8].copy_from_slice(
                &super::encode_direction([1., 0., 0.], -1.)
                    .unwrap()
                    .to_le_bytes(),
            );
        }
        for (index, value) in [0_u16, 1, 2, 0, 2, 1].into_iter().enumerate() {
            bytes[96 + index * 2..98 + index * 2].copy_from_slice(&value.to_le_bytes());
        }
        let chunk = |lod, position, normal, indices| {
            serde_json::json!({"lodMask":lod,"numVertices":3,"numIndices":3,
            "chunkVertices":{"byteOffsets":[position,normal],"vertexLayout":{"slotStrides":[8,8],"elements":[{"usage":"PS_Position"},{"usage":"PS_Normal","streamIndex":1},{"usage":"PS_Tangent","streamIndex":1}]}},"chunkIndices":{"teOffset":indices}})
        };
        let mut document = serde_json::json!({"Data":{"RootChunk":{"appearances":[],"renderResourceBlob":{"Data":{
            "header":{"quantizationScale":{"X":1.,"Y":1.,"Z":1.},"quantizationOffset":{"X":0.,"Y":0.,"Z":0.},"indexBufferOffset":96,"renderChunkInfos":[chunk(1,0,48,0),chunk(2,24,72,6)]},
            "renderBuffer":{"Bytes":base64::engine::general_purpose::STANDARD.encode(bytes)}}}}}});
        let mut replacement = super::decode_geometry(&document, false).unwrap();
        replacement[0].positions = vec![[-2., 0., 0.], [0., 3., 0.], [0., 0., 4.]];
        replacement[1].positions = vec![[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
        super::replace_geometry(&mut document, &replacement).unwrap();
        let decoded = super::decode_geometry(&document, false).unwrap();
        assert_eq!(decoded.len(), 2);
        for (actual, expected) in decoded.iter().zip(&replacement) {
            assert_eq!(actual.indices, expected.indices);
            assert_eq!(actual.chunk_index, expected.chunk_index);
            for (a, b) in actual.positions.iter().zip(&expected.positions) {
                for axis in 0..3 {
                    assert!((a[axis] - b[axis]).abs() < 0.001);
                }
            }
            assert!(actual.tangents.iter().all(|v| v[3] == -1.));
        }
        assert!(super::replace_geometry(&mut document, &replacement[..1]).is_err());
    }
}
