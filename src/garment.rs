//! Authored garment-support proxy geometry and deformation directions.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::Vec3;
use serde_json::Value;

use crate::assets::AssetError;

/// One native support proxy. Coordinates use metres with Y up, like mesh decoding.
pub struct SupportChunk {
    pub positions: Vec<Vec3>,
    pub offsets: Vec<Vec3>,
    /// Native flags are retained without reinterpreting undocumented bits.
    pub flags: Vec<u32>,
    pub cap_vertices: Vec<u32>,
    pub lod_mask: u64,
    pub custom_morph: bool,
}

/// Decodes the native support proxies without modifying the original mesh data.
///
/// # Errors
/// Returns an error for truncated/nonfinite vertex buffers or mismatched streams.
pub fn decode(document: &Value) -> Result<Vec<SupportChunk>, AssetError> {
    let parameters = document["Data"]["RootChunk"]["parameters"].as_array();
    let entries: Vec<_> = parameters
        .into_iter()
        .flatten()
        .map(|p| p.get("Data").unwrap_or(p))
        .collect();
    let caps = entries
        .iter()
        .find(|p| p["$type"] == "meshMeshParamGarmentSupport");
    let Some(garment) = entries
        .iter()
        .find(|p| p["$type"] == "garmentMeshParamGarment")
    else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for (index, chunk) in garment["chunks"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let positions = vectors(&chunk["vertices"])?;
        let mut offsets = vectors(&chunk["morphOffsets"])?;
        if offsets.is_empty() {
            offsets.resize(positions.len(), Vec3::ZERO);
        }
        let mut bytes = STANDARD.decode(chunk["garmentFlags"]["Bytes"].as_str().unwrap_or(""))?;
        if bytes.is_empty() {
            bytes.resize(positions.len() * 4, 0);
        }
        if positions.len() != offsets.len() || bytes.len() != positions.len() * 4 {
            return Err(AssetError::Invalid(
                "garment support stream counts differ".into(),
            ));
        }
        result.push(SupportChunk {
            positions,
            offsets,
            flags: bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            cap_vertices: caps
                .and_then(|p| p["chunkCapVertices"][index].as_array())
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
                .filter_map(|v| u32::try_from(v).ok())
                .collect(),
            lod_mask: chunk["lodMask"].as_u64().unwrap_or(1),
            custom_morph: caps.is_some_and(|p| p["customMorph"] == true),
        });
    }
    Ok(result)
}

fn vectors(buffer: &Value) -> Result<Vec<Vec3>, AssetError> {
    let bytes = STANDARD.decode(buffer["Bytes"].as_str().unwrap_or(""))?;
    if bytes.len() % 12 != 0 {
        return Err(AssetError::Invalid(
            "truncated garment vector buffer".into(),
        ));
    }
    bytes
        .chunks_exact(12)
        .map(|v| {
            let read = |offset| {
                f32::from_le_bytes([v[offset], v[offset + 1], v[offset + 2], v[offset + 3]])
            };
            let point = Vec3::new(read(0), read(8), -read(4));
            if point.is_finite() {
                Ok(point)
            } else {
                Err(AssetError::Invalid(
                    "nonfinite garment support vector".into(),
                ))
            }
        })
        .collect()
}
