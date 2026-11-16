//! Sparse geometry deltas and embedded base geometry from native morphtargets.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;

use crate::{
    assets::{AssetError, ResourceKey},
    mesh::{self, MeshChunk, MeshSkin},
};

/// Sparse deltas for one render chunk, in renderer Y-up coordinates.
#[derive(Clone, Debug)]
pub struct MorphChunk {
    /// Original native render-chunk index.
    pub chunk: usize,
    /// Vertex indices affected by this target.
    pub vertices: Vec<u16>,
    /// Position deltas in metres.
    pub positions: Vec<[f32; 3]>,
    /// Normal deltas, applied before normalization.
    pub normals: Vec<[f32; 3]>,
    /// Tangent deltas, applied before normalization.
    pub tangents: Vec<[f32; 3]>,
}

/// One named native morph shape.
#[derive(Clone, Debug)]
pub struct MorphTarget {
    /// Native target name.
    pub name: String,
    /// Native target region.
    pub region: String,
    /// Sparse changes by original render chunk.
    pub chunks: Vec<MorphChunk>,
}

/// Decoded morphtarget geometry, palette and sparse shapes.
#[derive(Debug)]
pub struct MorphMesh {
    /// Material and palette source mesh.
    pub base_mesh: ResourceKey,
    /// Embedded morphtarget base geometry, which overrides the source mesh shape.
    pub chunks: Vec<MeshChunk>,
    /// Named mesh bone palette.
    pub skin: Option<MeshSkin>,
    /// Shapes in authored order.
    pub targets: Vec<MorphTarget>,
}

/// Returns the material source mesh referenced by a morphtarget resource.
///
/// # Errors
/// Returns [`AssetError`] for missing native resource references.
pub fn base_mesh(document: &Value) -> Result<ResourceKey, AssetError> {
    ResourceKey::from_value(&document["Data"]["RootChunk"]["baseMesh"])?
        .ok_or_else(|| invalid("morphtarget has no base mesh"))
}

/// Decodes embedded geometry and packed position/normal/tangent deltas.
///
/// `base` is the decoded `baseMesh` resource and supplies its materials and bone
/// palette. Geometry comes from the morphtarget's embedded base blob.
///
/// # Errors
/// Returns [`AssetError`] for malformed layouts, truncated buffers or invalid mappings.
pub fn decode(document: &Value, base: &Value) -> Result<MorphMesh, AssetError> {
    decode_lods(document, base, true)
}

fn decode_lods(document: &Value, base: &Value, lod_filter: bool) -> Result<MorphMesh, AssetError> {
    let root = &document["Data"]["RootChunk"];
    if root["$type"] != "MorphTargetMesh" {
        return Err(invalid("expected MorphTargetMesh"));
    }
    let blob = &root["blob"]["Data"];
    let header = &blob["header"];
    let mut geometry = base.clone();
    geometry["Data"]["RootChunk"]["renderResourceBlob"] = blob["baseBlob"].clone();
    let chunks =
        mesh::decode_geometry(&geometry, lod_filter).map_err(|e| invalid(e.to_string()))?;
    let skin = mesh::decode_skin(base).map_err(|e| invalid(e.to_string()))?;
    let diffs = bytes(&blob["diffsBuffer"])?;
    let mappings = bytes(&blob["mappingBuffer"])?;
    let definitions = array(&root["targets"])?;
    if definitions.len() > 4096 {
        return Err(invalid("morphtarget target count exceeds 4096"));
    }
    let declared = integer(&header["numTargets"])?;
    if declared != 0 && declared != definitions.len() {
        return Err(invalid("morph target name/count mismatch"));
    }
    let mut targets = Vec::with_capacity(definitions.len());
    for (target, definition) in definitions.iter().enumerate() {
        let counts = array(&header["numVertexDiffsInEachChunk"][target])?;
        let map_counts = array(&header["numVertexDiffsMappingInEachChunk"][target])?;
        if counts.len() != map_counts.len() {
            return Err(invalid("morph chunk/map counts differ"));
        }
        let mut diff_offset = integer(&header["targetStartsInVertexDiffs"][target])?
            .checked_mul(12)
            .ok_or_else(|| invalid("morph diff offset overflow"))?;
        let mut map_offset = integer(&header["targetStartsInVertexDiffsMapping"][target])?
            .checked_mul(4)
            .ok_or_else(|| invalid("morph map offset overflow"))?;
        let empty = counts
            .iter()
            .map(integer)
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .all(|n| *n == 0);
        // Native empty targets retain unused infinity/NaN quantization sentinels.
        // They have no deltas to decode; nonempty targets still require finite data.
        let (offset, scale) = target_quantization(header, target, empty)?;
        let mut changes = Vec::new();
        for (chunk, count) in counts.iter().enumerate() {
            let count = integer(count)?;
            let mapping_bytes = integer(&map_counts[chunk])?
                .checked_mul(4)
                .ok_or_else(|| invalid("morph mapping length overflow"))?;
            if mapping_bytes / 2 != count + count % 2 {
                return Err(invalid("morph mapping count differs from delta count"));
            }
            let diff_bytes = count
                .checked_mul(12)
                .ok_or_else(|| invalid("morph delta length overflow"))?;
            let data = slice(&diffs, diff_offset, diff_bytes)?;
            let maps = slice(&mappings, map_offset, mapping_bytes)?;
            if let Some(mesh) = chunks.iter().find(|c| c.chunk_index == chunk) {
                let mut change = MorphChunk {
                    chunk,
                    vertices: Vec::with_capacity(count),
                    positions: Vec::with_capacity(count),
                    normals: Vec::with_capacity(count),
                    tangents: Vec::with_capacity(count),
                };
                for index in 0..count {
                    let vertex = u16::from_le_bytes([maps[index * 2], maps[index * 2 + 1]]);
                    if usize::from(vertex) >= mesh.positions.len() {
                        return Err(invalid("morph mapping exceeds vertex count"));
                    }
                    let position = unpack(data, index * 12, false)?;
                    let position = std::array::from_fn(|i| position[i] * scale[i] + offset[i]);
                    change.vertices.push(vertex);
                    change.positions.push(convert(position));
                    change
                        .normals
                        .push(convert(unpack(data, index * 12 + 4, true)?));
                    change
                        .tangents
                        .push(convert(unpack(data, index * 12 + 8, true)?));
                }
                changes.push(change);
            }
            diff_offset = diff_offset
                .checked_add(diff_bytes)
                .ok_or_else(|| invalid("morph diff offset overflow"))?;
            map_offset = map_offset
                .checked_add(mapping_bytes)
                .ok_or_else(|| invalid("morph map offset overflow"))?;
        }
        targets.push(MorphTarget {
            name: text(&definition["name"])?.to_owned(),
            region: text(&definition["regionName"])?.to_owned(),
            chunks: changes,
        });
    }
    Ok(MorphMesh {
        base_mesh: base_mesh(document)?,
        chunks,
        skin,
        targets,
    })
}

/// Bakes named native morph weights into a standalone mesh document for export.
///
/// Every LOD is deformed. Skin weights, materials and non-geometric streams come
/// from the base mesh; vertex geometry comes from the morph's embedded base blob,
/// matching live renderer deformation rather than substituting a different base.
///
/// # Errors
/// Returns an error for unknown/nonfinite weights, invalid sparse mappings or
/// geometry that cannot be encoded in the native layout.
pub fn bake(
    document: &Value,
    base: &Value,
    weights: &std::collections::BTreeMap<String, f32>,
) -> Result<Value, AssetError> {
    let mut morph = decode_lods(document, base, false)?;
    for (name, weight) in weights {
        if !weight.is_finite() {
            return Err(invalid("morph weight must be finite"));
        }
        let target = morph
            .targets
            .iter()
            .find(|t| t.name == *name)
            .ok_or_else(|| invalid(format!("unknown morph target {name}")))?;
        for change in &target.chunks {
            let chunk = morph
                .chunks
                .iter_mut()
                .find(|c| c.chunk_index == change.chunk)
                .ok_or_else(|| invalid("morph references an absent chunk"))?;
            for (i, vertex) in change.vertices.iter().enumerate() {
                let index = usize::from(*vertex);
                for axis in 0..3 {
                    chunk.positions[index][axis] += change.positions[i][axis] * weight;
                    if let Some(normal) = chunk.normals.get_mut(index) {
                        normal[axis] += change.normals[i][axis] * weight;
                    }
                    if let Some(tangent) = chunk.tangents.get_mut(index) {
                        tangent[axis] += change.tangents[i][axis] * weight;
                    }
                }
            }
        }
    }
    let mut output = base.clone();
    let mut blob = document["Data"]["RootChunk"]["blob"]["Data"]["baseBlob"].clone();
    // The embedded morph blob comes from a different CR2W container. Its buffer
    // identity must refer to the destination mesh's native render buffer slot.
    if let Some(id) = base.pointer("/Data/RootChunk/renderResourceBlob/Data/renderBuffer/BufferId")
    {
        blob["Data"]["renderBuffer"]["BufferId"] = id.clone();
    }
    rebase_blob_handles(&mut blob, &output)?;
    output["Data"]["RootChunk"]["renderResourceBlob"] = blob;
    mesh::replace_geometry(&mut output, &morph.chunks).map_err(|e| invalid(e.to_string()))?;
    Ok(output)
}

// Handle identities are local to each source document. Moving the embedded
// morph geometry into its material-source mesh must not alias material handles.
fn rebase_blob_handles(blob: &mut Value, base: &Value) -> Result<(), AssetError> {
    fn visit(value: &Value, ids: &mut std::collections::BTreeSet<String>) {
        match value {
            Value::Object(fields) => {
                if let Some(id) = fields.get("HandleId").and_then(Value::as_str) {
                    ids.insert(id.into());
                }
                for value in fields.values() {
                    visit(value, ids);
                }
            }
            Value::Array(values) => {
                for value in values {
                    visit(value, ids);
                }
            }
            _ => {}
        }
    }
    fn rewrite(value: &mut Value, mapping: &std::collections::BTreeMap<String, String>) {
        match value {
            Value::Object(fields) => {
                for key in ["HandleId", "HandleRefId"] {
                    if let Some(id) = fields
                        .get(key)
                        .and_then(Value::as_str)
                        .and_then(|id| mapping.get(id))
                        .cloned()
                    {
                        fields.insert(key.into(), Value::String(id));
                    }
                }
                for value in fields.values_mut() {
                    rewrite(value, mapping);
                }
            }
            Value::Array(values) => {
                for value in values {
                    rewrite(value, mapping);
                }
            }
            _ => {}
        }
    }
    let mut used = std::collections::BTreeSet::new();
    visit(base, &mut used);
    let mut next = used
        .iter()
        .filter_map(|id| id.parse::<u64>().ok())
        .max()
        .unwrap_or(0);
    let mut ids = std::collections::BTreeSet::new();
    visit(blob, &mut ids);
    let mut mapping = std::collections::BTreeMap::new();
    for id in ids {
        next = next
            .checked_add(1)
            .ok_or_else(|| invalid("native handle identity overflow"))?;
        mapping.insert(id, next.to_string());
    }
    rewrite(blob, &mapping);
    Ok(())
}

fn target_quantization(
    header: &Value,
    target: usize,
    empty: bool,
) -> Result<([f32; 3], [f32; 3]), AssetError> {
    if empty {
        return Ok(([0.; 3], [0.; 3]));
    }
    Ok((
        vector(&header["targetPositionDiffOffset"][target])?,
        vector(&header["targetPositionDiffScale"][target])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unused_empty_target_sentinels_do_not_weaken_nonempty_validation() {
        let header = serde_json::json!({"targetPositionDiffOffset":[{"X":f64::from(f32::MAX),"Y":f64::from(f32::MAX),"Z":f64::from(f32::MAX)}],"targetPositionDiffScale":[{"X":null,"Y":null,"Z":null}]});
        assert_eq!(
            target_quantization(&header, 0, true).unwrap(),
            ([0.; 3], [0.; 3])
        );
        assert!(target_quantization(&header, 0, false).is_err());
    }
}

fn invalid(message: impl Into<String>) -> AssetError {
    AssetError::Invalid(message.into())
}
fn text(value: &Value) -> Result<&str, AssetError> {
    value
        .get("$value")
        .unwrap_or(value)
        .as_str()
        .ok_or_else(|| invalid("missing morph name"))
}
fn array(value: &Value) -> Result<&[Value], AssetError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("missing morph array"))
}
fn integer(value: &Value) -> Result<usize, AssetError> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| invalid("invalid morph count/offset"))
}
fn bytes(value: &Value) -> Result<Vec<u8>, AssetError> {
    Ok(STANDARD.decode(
        value["Bytes"]
            .as_str()
            .ok_or_else(|| invalid("missing morph buffer bytes"))?,
    )?)
}
fn slice(bytes: &[u8], offset: usize, len: usize) -> Result<&[u8], AssetError> {
    bytes
        .get(
            offset
                ..offset
                    .checked_add(len)
                    .ok_or_else(|| invalid("morph buffer length overflow"))?,
        )
        .ok_or_else(|| invalid("truncated morph buffer"))
}
fn vector(value: &Value) -> Result<[f32; 3], AssetError> {
    super::assets::vector(Some(value), [0.; 3])
}
fn convert(value: [f32; 3]) -> [f32; 3] {
    [value[0], value[2], -value[1]]
}
fn unpack(bytes: &[u8], offset: usize, signed: bool) -> Result<[f32; 3], AssetError> {
    let packed = u32::from_le_bytes(
        slice(bytes, offset, 4)?
            .try_into()
            .map_err(|_| invalid("invalid packed morph delta"))?,
    );
    Ok(std::array::from_fn(|i| {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ten-bit packed values are exactly representable"
        )]
        let value = ((packed >> (i * 10)) & 1023) as f32 / 1023.0;
        if signed { value * 2.0 - 1.0 } else { value }
    }))
}
