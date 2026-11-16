//! Native world collision primitives, independent of editor placement rules.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// Invalid or unsupported native collision data.
#[derive(Debug, Error)]
pub enum WorldError {
    /// A required native field or bounded buffer span is absent.
    #[error("invalid native world data: {0}")]
    Invalid(String),
    /// A native base64 buffer cannot be decoded.
    #[error("invalid native world buffer: {0}")]
    Base64(#[from] base64::DecodeError),
}
fn bad(message: &str) -> WorldError {
    WorldError::Invalid(message.into())
}

fn cooked_count(bytes: &[u8], cursor: &mut usize) -> Result<usize, WorldError> {
    let first = *bytes
        .get(*cursor)
        .ok_or_else(|| bad("cooked count absent"))?;
    *cursor += 1;
    if first & 0x80 != 0 {
        return Err(bad("negative cooked count"));
    }
    let mut value = usize::from(first & 0x3f);
    if first & 0x40 != 0 {
        let mut shift = 6;
        loop {
            let next = *bytes
                .get(*cursor)
                .ok_or_else(|| bad("truncated cooked count"))?;
            *cursor += 1;
            value |= usize::from(next & 0x7f) << shift;
            if next & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 27 {
                return Err(bad("cooked count overflow"));
            }
        }
    }
    Ok(value)
}

/// Resolves the per-placement record of a 2.5 cooked world node while retaining
/// its native class and the original appendix. Legacy nodes retain their fields.
/// Layouts follow WolvenKit's `update_2.5` WorldCooked readers.
///
/// # Errors
/// Rejects absent placement indices, truncated descriptors and transform ranges.
pub fn placement_node(node: &Value, placement: &Value) -> Result<Value, WorldError> {
    let kind = node["$type"].as_str().unwrap_or("");
    let instanced =
        kind == "worldInstancedMeshNode" && node.get("cookedWorldTransformsBuffer").is_some();
    let stride = match kind {
        "worldCookedStaticMeshNode" => 33,
        "worldCookedGenericProxyMeshNode" => 41,
        "worldCookedCollisionNode" => 48,
        _ if instanced => 64,
        _ => return Ok(node.clone()),
    };
    let index = placement["InstanceIndex"]
        .as_u64()
        .and_then(|i| usize::try_from(i).ok())
        .ok_or_else(|| bad("cooked placement index absent"))?;
    let tail = STANDARD.decode(
        node["$rawTail"]
            .as_str()
            .ok_or_else(|| bad("cooked appendix absent"))?,
    )?;
    let mut cursor = usize::from(instanced);
    let length = cooked_count(&tail, &mut cursor)?;
    let count = if instanced {
        length
    } else {
        if length % stride != 0 {
            return Err(bad("cooked descriptor length"));
        }
        length / stride
    };
    if index >= count || count > tail.len() / stride {
        return Err(bad("cooked placement outside descriptors"));
    }
    let end = cursor
        .checked_add(count * stride)
        .filter(|end| *end <= tail.len())
        .ok_or_else(|| bad("truncated cooked descriptors"))?;
    let record = &tail[cursor + index * stride..cursor + (index + 1) * stride];
    let mut resolved = node.clone();
    if kind == "worldCookedCollisionNode" {
        resolved["sectorHash"] = json!(u64::from_le_bytes(take(record, 16)?).to_string());
        let offset = u32::from_le_bytes(take(record, 24)?) as usize;
        for (i, name) in [
            "numActors",
            "numShapeInfos",
            "numShapePositions",
            "numShapeRotations",
            "numScales",
            "numMaterials",
            "numPresets",
            "numShapeIndices",
            "numMaterialIndices",
        ]
        .iter()
        .enumerate()
        {
            resolved[*name] = json!(u16::from_le_bytes(take(record, 28 + i * 2)?));
        }
        let data = STANDARD.decode(
            node["compiledData"]["Bytes"]
                .as_str()
                .ok_or_else(|| bad("cooked collision data absent"))?,
        )?;
        resolved["compiledData"]["Bytes"] = json!(
            STANDARD.encode(
                data.get(offset..)
                    .ok_or_else(|| bad("cooked collision offset outside buffer"))?
            )
        );
        return Ok(resolved);
    }
    let mesh_offset = if instanced { 32 } else { 0 };
    resolved["mesh"] = json!({"DepotPath":{"$type":"ResourcePath","$storage":"uint64",
        "$value":u64::from_le_bytes(take(record, mesh_offset)?).to_string()},"Flags":"Default"});
    resolved["meshAppearance"] = json!({"$type":"CName","$storage":"uint64",
        "$value":u64::from_le_bytes(take(record, mesh_offset + 8)?).to_string()});
    if instanced {
        cursor = end;
        let ranges = cooked_count(&tail, &mut cursor)?;
        if ranges != count || cursor.checked_add(ranges * 4) != Some(tail.len()) {
            return Err(bad("cooked transform range table"));
        }
        let start = usize::from(u16::from_le_bytes(take(&tail, cursor + index * 4)?));
        let count = usize::from(u16::from_le_bytes(take(&tail, cursor + index * 4 + 2)?));
        let buffer = &node["cookedWorldTransformsBuffer"];
        let bytes = STANDARD.decode(
            buffer["Bytes"]
                .as_str()
                .ok_or_else(|| bad("cooked transforms absent"))?,
        )?;
        if (start + count)
            .checked_mul(48)
            .is_none_or(|end| end > bytes.len())
        {
            return Err(bad("cooked transforms truncated"));
        }
        resolved["worldTransformsBuffer"] = json!({"startIndex":start,"numElements":count,
            "sharedDataBuffer":{"Data":{"buffer":buffer}}});
    } else if end != tail.len() {
        return Err(bad("cooked descriptor trailing bytes"));
    }
    Ok(resolved)
}
fn take<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], WorldError> {
    bytes
        .get(offset..offset + N)
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| bad("truncated collision buffer"))
}
fn count(node: &Value, key: &str) -> Result<usize, WorldError> {
    usize::try_from(node[key].as_u64().unwrap_or(0))
        .map_err(|_| bad("native count exceeds platform range"))
}
fn short(bytes: &[u8], offset: usize) -> Result<i16, WorldError> {
    Ok(i16::from_le_bytes(take(bytes, offset)?))
}
fn scalar(bytes: &[u8], offset: usize) -> Result<f32, WorldError> {
    let value = f32::from_le_bytes(take(bytes, offset)?);
    if !value.is_finite() {
        return Err(bad("nonfinite collision transform"));
    }
    Ok(value)
}
fn vector(bytes: &[u8], offset: usize) -> Result<Vec3, WorldError> {
    Ok(Vec3::new(
        scalar(bytes, offset)?,
        scalar(bytes, offset + 4)?,
        scalar(bytes, offset + 8)?,
    ))
}
fn rotation(bytes: &[u8], offset: usize) -> Result<Quat, WorldError> {
    let q = Quat::from_xyzw(
        scalar(bytes, offset)?,
        scalar(bytes, offset + 4)?,
        scalar(bytes, offset + 8)?,
        scalar(bytes, offset + 12)?,
    );
    if q.length_squared() < 0.00001 {
        return Err(bad("zero collision quaternion"));
    }
    Ok(q.normalize())
}

/// One actor-relative native collision shape in world coordinates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionShape {
    /// Native per-face physics material hashes.
    pub materials: Vec<u64>,
    /// Native filter-preset hash.
    pub preset: u64,
    /// Native proxy discriminator.
    pub proxy: u8,
    /// Actor index within the node's actor table.
    pub actor: usize,
    /// Shape index relative to this actor.
    pub shape: usize,
    /// Native `PhysX` shape discriminator; zero denotes a box.
    pub kind: u16,
    /// World-space shape centre in RED coordinates.
    pub position: [f32; 3],
    /// World-space quaternion, x/y/z/w.
    pub rotation: [f32; 4],
    /// Native scale, including actor scale.
    pub scale: [f32; 3],
    /// Unscaled half-extents for a primitive box.
    pub half_extents: Option<[f32; 3]>,
    /// Native cached collision geometry identity for mesh/convex shapes.
    pub geometry: Option<u64>,
}

/// Decodes actor, scale and shape-relative transforms without losing individual shapes.
///
/// # Errors
/// Rejects malformed table counts, transform indices and nonfinite/zero rotations.
#[expect(
    clippy::cast_precision_loss,
    reason = "native fixed-point coordinates are intentionally converted to game f32 metres"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one bounded decoder keeps the native actor/shape table layout together"
)]
pub fn collision_shapes(node: &Value) -> Result<Vec<CollisionShape>, WorldError> {
    let bytes = STANDARD.decode(
        node["compiledData"]["Bytes"]
            .as_str()
            .ok_or_else(|| bad("collision bytes absent"))?,
    )?;
    decode_collision_shapes(node, &bytes)
}

fn decode_collision_shapes(node: &Value, bytes: &[u8]) -> Result<Vec<CollisionShape>, WorldError> {
    let actors = count(node, "numActors")?;
    let shapes = count(node, "numShapeInfos")?;
    let positions = count(node, "numShapePositions")?;
    let rotations = count(node, "numShapeRotations")?;
    let scales = count(node, "numScales")?;
    if actors > 1_000_000
        || shapes > 4_000_000
        || positions > 4_000_000
        || rotations > 4_000_000
        || scales > 1_000_000
    {
        return Err(bad("collision tables exceed bounded limits"));
    }
    let position_start = actors * 48 + shapes * 32;
    let rotation_start = position_start + positions * 12;
    let scale_start = rotation_start + rotations * 16;
    if scale_start + scales * 12 > bytes.len() {
        return Err(bad("collision tables truncated"));
    }
    let material_start = scale_start + scales * 12;
    let preset_start = material_start + count(node, "numMaterials")? * 8;
    let material_indices =
        preset_start + count(node, "numPresets")? * 8 + count(node, "numShapeIndices")? * 4;
    let mut output = Vec::new();
    for actor in 0..actors {
        let off = actor * 48;
        let position = Vec3::new(
            i32::from_le_bytes(take(&bytes, off)?) as f32,
            i32::from_le_bytes(take(&bytes, off + 4)?) as f32,
            i32::from_le_bytes(take(&bytes, off + 8)?) as f32,
        ) / 131_072.;
        let orientation = rotation(&bytes, off + 16)?;
        let scale_index = short(&bytes, off + 36)?;
        let scale = if scales > 0 && scale_index >= 0 {
            let i = usize::try_from(scale_index).map_err(|_| bad("invalid scale index"))?;
            if i >= scales {
                return Err(bad("scale index exceeds table"));
            }
            vector(&bytes, scale_start + i * 12)?
        } else {
            Vec3::ONE
        };
        let first = usize::from(u16::from_le_bytes(take(&bytes, off + 32)?));
        let size = usize::from(u16::from_le_bytes(take(&bytes, off + 34)?));
        if first + size > shapes {
            return Err(bad("actor shape range exceeds table"));
        }
        for shape in 0..size {
            let off = actors * 48 + (first + shape) * 32;
            let kind = u16::from_le_bytes(take(&bytes, off + 12)?);
            // Bytes 14..20 contain the unknown/material range, not transforms.
            let pi = short(&bytes, off + 20)?;
            let qi = short(&bytes, off + 22)?;
            let local = if positions > 0 && pi >= 0 {
                let i = usize::try_from(pi).map_err(|_| bad("invalid shape offset index"))?;
                if i >= positions {
                    return Err(bad("shape offset exceeds table"));
                }
                vector(&bytes, position_start + i * 12)?
            } else {
                Vec3::ZERO
            };
            let q = if rotations > 0 && qi >= 0 {
                let i = usize::try_from(qi).map_err(|_| bad("invalid shape rotation index"))?;
                if i >= rotations {
                    return Err(bad("shape rotation exceeds table"));
                }
                rotation(&bytes, rotation_start + i * 16)?
            } else {
                Quat::IDENTITY
            };
            output.push(CollisionShape {
                materials: {
                    let start = usize::try_from(short(&bytes, off + 16)?)
                        .map_err(|_| bad("negative material range"))?;
                    let count = usize::try_from(short(&bytes, off + 18)?)
                        .map_err(|_| bad("negative material count"))?;
                    (start..start + count)
                        .map(|i| {
                            let material = usize::from(
                                *bytes
                                    .get(material_indices + i)
                                    .ok_or_else(|| bad("material index table truncated"))?,
                            );
                            Ok(u64::from_le_bytes(take(
                                &bytes,
                                material_start + material * 8,
                            )?))
                        })
                        .collect::<Result<Vec<_>, WorldError>>()?
                },
                preset: u64::from_le_bytes(take(
                    &bytes,
                    preset_start
                        + usize::from(
                            *bytes
                                .get(off + 24)
                                .ok_or_else(|| bad("preset index absent"))?,
                        ) * 8,
                )?),
                proxy: *bytes.get(off + 25).ok_or_else(|| bad("proxy absent"))?,
                actor,
                shape,
                kind,
                position: (position + orientation * (local * scale)).to_array(),
                rotation: (orientation * q).to_array(),
                scale: scale.to_array(),
                half_extents: if kind == 0 {
                    Some(vector(&bytes, off)?.abs().to_array())
                } else {
                    None
                },
                geometry: if kind == 3 || kind == 4 {
                    Some(u64::from_le_bytes(take(&bytes, off)?))
                } else {
                    None
                },
            });
        }
    }
    Ok(output)
}

/// One native collision data group and its geometry-cache sector identity.
#[derive(Debug)]
pub struct CollisionGroup {
    /// Identity used when resolving this group's cooked geometry cache entries.
    pub sector_hash: u64,
    /// Actor-relative shapes decoded into world coordinates.
    pub shapes: Vec<CollisionShape>,
}

/// Decodes terrain heightfields into world-space collision triangles, preserving holes.
///
/// # Errors
/// Rejects malformed transforms, scales or cooked heightfield data.
pub fn terrain_collision(node: &Value) -> Result<crate::physics::CollisionGeometry, WorldError> {
    let buffer = STANDARD.decode(
        node["heightfieldGeometry"]["Bytes"]
            .as_str()
            .ok_or_else(|| bad("terrain heightfield absent"))?,
    )?;
    let field = crate::physics::decode_heightfield(&buffer)?;
    let t = &node["actorTransform"];
    let mut p = [0f32; 3];
    for (i, key) in ["x", "y", "z"].iter().enumerate() {
        p[i] = t["Position"][key]["Bits"]
            .as_i64()
            .ok_or_else(|| bad("terrain position absent"))? as f32
            / 131072.;
    }
    let q = &t["Orientation"];
    let rotation = Quat::from_xyzw(
        q["i"].as_f64().unwrap_or(0.) as f32,
        q["j"].as_f64().unwrap_or(0.) as f32,
        q["k"].as_f64().unwrap_or(0.) as f32,
        q["r"].as_f64().unwrap_or(1.) as f32,
    );
    let row = node["rowScale"]
        .as_f64()
        .ok_or_else(|| bad("terrain row scale absent"))? as f32;
    let col = node["columnScale"]
        .as_f64()
        .ok_or_else(|| bad("terrain column scale absent"))? as f32;
    let height = node["heightScale"]
        .as_f64()
        .ok_or_else(|| bad("terrain height scale absent"))? as f32;
    if !row.is_finite()
        || !col.is_finite()
        || !height.is_finite()
        || row <= 0.
        || col <= 0.
        || height <= 0.
        || !rotation.is_finite()
        || rotation.length_squared() < 1e-6
    {
        return Err(bad("invalid terrain transform"));
    }
    let rotation = rotation.normalize();
    let p = Vec3::from_array(p);
    let vertices = field
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (p + rotation
                * Vec3::new(
                    (i / field.columns) as f32 * row,
                    f32::from(s.height) * height,
                    (i % field.columns) as f32 * col,
                ))
            .to_array()
        })
        .collect::<Vec<_>>();
    let mut triangles = Vec::new();
    let mut materials = Vec::new();
    for r in 0..field.rows - 1 {
        for c in 0..field.columns - 1 {
            let i = r * field.columns + c;
            let cols = field.columns;
            let s = &field.samples[i];
            let faces = if s.materials[0] & 0x80 != 0 {
                [[i + cols, i, i + cols + 1], [i + 1, i + cols + 1, i]]
            } else {
                [[i, i + 1, i + cols], [i + cols + 1, i + cols, i + 1]]
            };
            for (face, material) in faces.into_iter().zip(s.materials) {
                if material & 0x7f != 0x7f {
                    triangles.push(face.map(|i| i as u32));
                    materials.push(u16::from(material & 0x7f));
                }
            }
        }
    }
    Ok(crate::physics::CollisionGeometry {
        convex: false,
        vertices,
        triangles,
        materials,
    })
}

/// Decodes ordinary collision nodes and cooked nodes containing multiple inner nodes.
/// The cooked 48-byte descriptor layout follows RED4ext SDK's
/// `worldCookedCollisionNode.hpp`; actor/shape tables use the same native layout.
///
/// # Errors
/// Rejects malformed cooked descriptor lengths, offsets and collision tables.
pub fn collision_groups(node: &Value) -> Result<Vec<CollisionGroup>, WorldError> {
    if node["$type"] != "worldCookedCollisionNode" {
        let sector_hash = node["sectorHash"]
            .as_u64()
            .or_else(|| node["sectorHash"].as_str()?.parse().ok())
            .ok_or_else(|| bad("collision sector hash absent"))?;
        return Ok(vec![CollisionGroup {
            sector_hash,
            shapes: collision_shapes(node)?,
        }]);
    }
    let tail = STANDARD.decode(
        node["$rawTail"]
            .as_str()
            .ok_or_else(|| bad("cooked collision descriptors absent"))?,
    )?;
    let first = *tail
        .first()
        .ok_or_else(|| bad("cooked collision count absent"))?;
    if first & 0x80 != 0 {
        return Err(bad("negative cooked collision size"));
    }
    let mut size = usize::from(first & 0x3f);
    let mut at = 1;
    if first & 0x40 != 0 {
        let mut shift = 6;
        loop {
            let b = *tail
                .get(at)
                .ok_or_else(|| bad("truncated cooked collision size"))?;
            at += 1;
            size |= usize::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 27 {
                return Err(bad("cooked collision size overflow"));
            }
        }
    }
    if size % 48 != 0 || at.checked_add(size) != Some(tail.len()) {
        return Err(bad("cooked collision descriptor span"));
    }
    let data = STANDARD.decode(
        node["compiledData"]["Bytes"]
            .as_str()
            .ok_or_else(|| bad("cooked collision data absent"))?,
    )?;
    tail[at..]
        .chunks_exact(48)
        .map(|d| {
            let sector_hash = u64::from_le_bytes(take(d, 16)?);
            let offset = usize::try_from(u32::from_le_bytes(take(d, 24)?))
                .map_err(|_| bad("cooked collision offset overflow"))?;
            let mut counts = json!({});
            for (i, name) in [
                "numActors",
                "numShapeInfos",
                "numShapePositions",
                "numShapeRotations",
                "numScales",
                "numMaterials",
                "numPresets",
                "numShapeIndices",
                "numMaterialIndices",
            ]
            .iter()
            .enumerate()
            {
                counts[*name] = json!(u16::from_le_bytes(take(d, 28 + i * 2)?));
            }
            Ok(CollisionGroup {
                sector_hash,
                shapes: decode_collision_shapes(
                    &counts,
                    data.get(offset..)
                        .ok_or_else(|| bad("cooked collision offset outside data"))?,
                )?,
            })
        })
        .collect()
}

/// A native world-space box for generated structural collision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionBox {
    /// World-space RED centre.
    pub position: [f32; 3],
    /// World-space x/y/z/w orientation.
    pub rotation: [f32; 4],
    /// Positive dimensions in metres.
    pub size: [f32; 3],
}

/// Encodes independent native static box actors without affecting source actor tables.
///
/// # Errors
/// Rejects invalid dimensions, transforms and coordinates outside native fixed-point range.
#[expect(
    clippy::cast_possible_truncation,
    reason = "fixed-point encoding is range checked before conversion"
)]
pub fn encode_boxes(boxes: &[CollisionBox], name: &str) -> Result<Value, WorldError> {
    if boxes.len() != 1 {
        return Err(bad(
            "encode one independent collision actor per native node",
        ));
    }
    let mut bytes = Vec::new();
    for (index, b) in boxes.iter().enumerate() {
        if b.size.iter().any(|v| !v.is_finite() || *v <= 0.)
            || b.rotation.iter().any(|v| !v.is_finite())
            || (Quat::from_array(b.rotation).length_squared() - 1.).abs() > 0.001
        {
            return Err(bad("invalid box dimensions or orientation"));
        }
        for p in b.position {
            let fixed = (f64::from(p) * 131_072.).round();
            if !fixed.is_finite() || fixed < f64::from(i32::MIN) || fixed > f64::from(i32::MAX) {
                return Err(bad("box position outside fixed-point range"));
            }
            bytes.extend((fixed as i32).to_le_bytes());
        }
        bytes.extend(0_i32.to_le_bytes());
        for q in b.rotation {
            bytes.extend(q.to_le_bytes());
        }
        bytes.extend(
            u16::try_from(index)
                .map_err(|_| bad("box actor index overflow"))?
                .to_le_bytes(),
        );
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend((-1_i16).to_le_bytes());
        bytes.extend(0_i16.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
    }
    for b in boxes {
        for v in b.size {
            bytes.extend((v / 2.).to_le_bytes());
        }
        for v in [0_i16, 0, 0, 1, -1, -1] {
            bytes.extend(v.to_le_bytes());
        }
        bytes.extend([0, 11]);
        bytes.extend(0_u16.to_le_bytes());
        bytes.extend(0_u32.to_le_bytes());
    }
    for value in ["concrete.physmat", "World Static"] {
        let hash = value.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        bytes.extend(hash.to_le_bytes());
    }
    bytes.extend(0_u32.to_le_bytes());
    for index in 0..boxes.len() {
        let mut value = u32::try_from(index).map_err(|_| bad("shape index overflow"))?;
        loop {
            let mut b = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                b |= 0x80;
            }
            bytes.push(b);
            if value == 0 {
                break;
            }
        }
    }
    let extent = boxes.iter().flat_map(|b| b.size).fold(0_f32, f32::max);
    Ok(
        json!({"$type":"worldCollisionNode","debugName":{"$type":"CName","$storage":"string","$value":name},"compiledData":{"BufferId":"0","Flags":4_063_232,"Bytes":STANDARD.encode(bytes)},"numActors":boxes.len(),"numShapeInfos":boxes.len(),"numShapePositions":0,"numShapeRotations":0,"numScales":0,"numMaterials":1,"numPresets":1,"numMaterialIndices":1,"numShapeIndices":boxes.len(),"sectorHash":"0","resourceVersion":2,"lod":0,"extents":{"$type":"Vector4","X":extent,"Y":extent,"Z":extent,"W":0.},"staticCollisionShapeCategories":{"$type":"worldStaticCollisionShapeCategories_CollisionNode","arr":{"Elements":[{"Elements":[0,0,0,0,0,0]},{"Elements":[0,1,0,0,0,0]},{"Elements":[0,0,0,0,0,0]},{"Elements":[0,0,0,0,0,0]},{"Elements":[0,1,0,0,0,0]}]}}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_offsets_follow_material_range_and_compose_with_actor_pose() {
        let b = CollisionBox {
            position: [10., 20., 3.],
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array(),
            size: [4., 0.2, 3.],
        };
        let mut node = encode_boxes(&[b], "offset shape").unwrap();
        let mut bytes = STANDARD
            .decode(node["compiledData"]["Bytes"].as_str().unwrap())
            .unwrap();
        bytes[48 + 20..48 + 22].copy_from_slice(&0_i16.to_le_bytes());
        bytes[48 + 22..48 + 24].copy_from_slice(&0_i16.to_le_bytes());
        let mut transform = Vec::new();
        for value in [1_f32, 0., 0.] {
            transform.extend(value.to_le_bytes());
        }
        for value in Quat::from_rotation_z(0.3).to_array() {
            transform.extend(value.to_le_bytes());
        }
        bytes.splice(80..80, transform);
        node["numShapePositions"] = json!(1);
        node["numShapeRotations"] = json!(1);
        node["compiledData"]["Bytes"] = json!(STANDARD.encode(bytes));
        let shapes = collision_shapes(&node).unwrap();
        assert!((Vec3::from_array(shapes[0].position) - Vec3::new(10., 21., 3.)).length() < 0.0001);
        assert!(
            Quat::from_array(shapes[0].rotation)
                .dot(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2 + 0.3))
                .abs()
                > 0.9999
        );
    }
    #[test]
    fn native_box_actor_transforms_round_trip() {
        let boxes = [CollisionBox {
            position: [4., -3., 2.],
            rotation: Quat::from_rotation_z(0.7).to_array(),
            size: [3., 0.2, 4.],
        }];
        let node = encode_boxes(&boxes, "test").unwrap();
        let decoded = collision_shapes(&node).unwrap();
        assert_eq!(decoded.len(), 1);
        for (b, s) in boxes.iter().zip(decoded) {
            assert!(
                (Vec3::from_array(b.position) - Vec3::from_array(s.position)).length() < 0.00001
            );
            assert!(
                (Vec3::from_array(b.size) / 2. - Vec3::from_array(s.half_extents.unwrap()))
                    .length()
                    < 0.00001
            );
        }
    }
}
