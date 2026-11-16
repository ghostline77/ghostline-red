use std::collections::HashSet;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::Mat4;
use serde_json::Value;

use super::{AssetError, Transform, invalid, root, text};

/// A named native bone with its local animation reference transform.
#[derive(Clone, Debug)]
pub struct Bone {
    /// Native name used by mesh skin palettes and attachment slots.
    pub name: String,
    /// Parent bone index; roots have no parent.
    pub parent: Option<usize>,
    /// Local animation reference transform in RED coordinates.
    /// Geometry binding may use the separate authored [`Rig::a_pose`].
    pub rest: Transform,
}

/// Bone hierarchy decoded from an `animRig` resource.
#[derive(Clone, Debug)]
pub struct Rig {
    /// Archive resource identity when loaded through an asset reader.
    pub source: Option<super::ResourceKey>,
    /// Bones in native index order.
    pub bones: Vec<Bone>,
    /// Authored local A-pose used to bind visible geometry, when present.
    /// Animation reference transforms remain in `bones[*].rest`.
    pub a_pose: Option<Vec<Transform>>,
}

impl Rig {
    /// Replaces the native local reference transforms, preserving hierarchy,
    /// names and appendix padding. An authored A-pose must be supplied separately
    /// when present. The document is unchanged on validation failure.
    ///
    /// # Errors
    /// Returns an error for invalid transforms or mismatched bone/A-pose counts.
    pub fn replace_reference_document(
        document: &mut Value,
        reference: &[Transform],
        a_pose: Option<&[Transform]>,
    ) -> Result<(), AssetError> {
        let original = Self::decode(document)?;
        if reference.len() != original.bones.len()
            || a_pose.is_some() != original.a_pose.is_some()
            || a_pose.is_some_and(|p| p.len() != reference.len())
        {
            return Err(invalid(
                "replacement rig pose count differs from native rig",
            ));
        }
        for transform in reference.iter().chain(a_pose.into_iter().flatten()) {
            transform.matrix()?;
        }
        let mut changed = document.clone();
        let root = if changed.get("Data").is_some() {
            &mut changed["Data"]["RootChunk"]
        } else {
            &mut changed
        };
        if let Some(tail) = root.get("$rawTail").and_then(Value::as_str) {
            let mut bytes = STANDARD.decode(tail)?;
            for (i, t) in reference.iter().enumerate() {
                let base = reference.len() * 2 + i * 48;
                for (start, values) in [
                    (0, t.translation.as_slice()),
                    (4, t.rotation.as_slice()),
                    (8, t.scale.as_slice()),
                ] {
                    for (axis, value) in values.iter().enumerate() {
                        let offset = base + (start + axis) * 4;
                        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            root["$rawTail"] = Value::String(STANDARD.encode(bytes));
        } else {
            for (pose, t) in root["boneTransforms"]
                .as_array_mut()
                .ok_or_else(|| invalid("rig reference transforms absent"))?
                .iter_mut()
                .zip(reference)
            {
                write_transform(pose, t);
            }
        }
        if let Some(a_pose) = a_pose {
            for (pose, t) in root["aPoseLS"]
                .as_array_mut()
                .ok_or_else(|| invalid("rig A-pose absent"))?
                .iter_mut()
                .zip(a_pose)
            {
                write_transform(pose, t);
            }
        }
        Self::decode(&changed)?.model_rest()?;
        *document = changed;
        Ok(())
    }
    /// Multiplies a native rig's first root scale while preserving the rest of
    /// its resource document and binary appendix. Child translations and their
    /// animation displacement inherit the scaled root.
    ///
    /// # Errors
    /// Returns an error for an invalid rig, non-root first bone, or nonpositive
    /// or nonfinite scale. The document is unchanged on failure.
    pub fn scale_root_document(document: &mut Value, scale: [f32; 3]) -> Result<(), AssetError> {
        if !scale.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(invalid("rig root scale must be finite and positive"));
        }
        let original = Self::decode(document)?;
        if original.bones[0].parent.is_some() {
            return Err(invalid("first rig bone is not a root"));
        }
        let mut changed = document.clone();
        let root = if changed.get("Data").is_some() {
            &mut changed["Data"]["RootChunk"]
        } else {
            &mut changed
        };
        if let Some(tail) = root.get("$rawTail").and_then(Value::as_str) {
            let mut bytes = STANDARD.decode(tail)?;
            // The appendix stores all i16 parent indices, then 48-byte QsTransforms.
            let offset = original.bones.len() * 2 + 32;
            for (axis, multiplier) in scale.into_iter().enumerate() {
                let value = original.bones[0].rest.scale[axis] * multiplier;
                bytes[offset + axis * 4..offset + axis * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
            root["$rawTail"] = Value::String(STANDARD.encode(bytes));
        } else {
            for (axis, multiplier) in ["X", "Y", "Z"].into_iter().zip(scale) {
                let value = root["boneTransforms"][0]["Scale"][axis]
                    .as_f64()
                    .ok_or_else(|| invalid("rig root scale component absent"))?;
                root["boneTransforms"][0]["Scale"][axis] =
                    serde_json::json!(value * f64::from(multiplier));
            }
        }
        if let Some(poses) = root
            .get_mut("aPoseLS")
            .and_then(Value::as_array_mut)
            .filter(|p| !p.is_empty())
        {
            for (axis, multiplier) in ["X", "Y", "Z"].into_iter().zip(scale) {
                let value = poses[0]["Scale"][axis]
                    .as_f64()
                    .ok_or_else(|| invalid("rig A-pose root scale component absent"))?;
                poses[0]["Scale"][axis] = serde_json::json!(value * f64::from(multiplier));
            }
        }
        Self::decode(&changed)?.model_rest()?;
        *document = changed;
        Ok(())
    }

    /// Decodes reflected fields or the native binary rig appendix.
    ///
    /// # Errors
    /// Returns [`AssetError`] for truncated appendices, invalid names, transforms
    /// or cyclic/out-of-range parent indices.
    pub fn decode(document: &Value) -> Result<Self, AssetError> {
        let root = root(document);
        if root.get("$type").and_then(Value::as_str) != Some("animRig") {
            return Err(invalid("expected animRig"));
        }
        let names = root
            .get("boneNames")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("rig has no bone names"))?;
        if names.is_empty() {
            return Err(invalid("rig is empty"));
        }
        let mut seen = HashSet::new();
        let mut bones = names
            .iter()
            .map(|value| {
                let name = text(value)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| invalid("empty rig bone name"))?;
                if !seen.insert(name) {
                    return Err(invalid(format!("duplicate rig bone {name}")));
                }
                Ok(Bone {
                    name: name.to_owned(),
                    parent: None,
                    rest: Transform::default(),
                })
            })
            .collect::<Result<Vec<_>, AssetError>>()?;
        if let Some(tail) = root.get("$rawTail").and_then(Value::as_str) {
            decode_appendix(&STANDARD.decode(tail)?, &mut bones)?;
        } else {
            let parents = root
                .get("boneParentIndexes")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("rig has no parent index data"))?;
            let transforms = root
                .get("boneTransforms")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("rig has no rest transforms"))?;
            if parents.len() != bones.len() || transforms.len() != bones.len() {
                return Err(invalid("rig bone/parent/transform counts differ"));
            }
            for (index, bone) in bones.iter_mut().enumerate() {
                bone.parent = parent_index(
                    parents[index]
                        .as_i64()
                        .ok_or_else(|| invalid("invalid rig parent index"))?,
                    names.len(),
                )?;
                bone.rest = Transform::decode(&transforms[index])?;
            }
        }
        let a_pose = root
            .get("aPoseLS")
            .and_then(Value::as_array)
            .filter(|p| !p.is_empty())
            .map(|poses| -> Result<Vec<Transform>, AssetError> {
                if poses.len() != bones.len() {
                    return Err(invalid("rig A-pose count differs from bone count"));
                }
                poses.iter().map(Transform::decode).collect()
            })
            .transpose()?;
        let result = Self {
            source: None,
            bones,
            a_pose,
        };
        result.model_rest()?;
        Ok(result)
    }

    /// Computes native model-space rest matrices in bone index order.
    ///
    /// # Errors
    /// Returns [`AssetError`] for invalid transforms or a malformed hierarchy.
    pub fn model_rest(&self) -> Result<Vec<Mat4>, AssetError> {
        let mut models = vec![Mat4::IDENTITY; self.bones.len()];
        let mut states = vec![0_u8; self.bones.len()];
        for index in 0..self.bones.len() {
            let mut path = Vec::new();
            let mut current = Some(index);
            while let Some(id) = current {
                let bone = self
                    .bones
                    .get(id)
                    .ok_or_else(|| invalid("rig parent index out of range"))?;
                if states[id] == 2 {
                    break;
                }
                if states[id] == 1 {
                    return Err(invalid("cyclic rig bone hierarchy"));
                }
                states[id] = 1;
                path.push(id);
                current = bone.parent;
            }
            for id in path.into_iter().rev() {
                let bone = &self.bones[id];
                models[id] =
                    bone.parent.map_or(Mat4::IDENTITY, |p| models[p]) * bone.rest.matrix()?;
                if !models[id].is_finite() || !models[id].inverse().is_finite() {
                    return Err(invalid("nonfinite or singular rig model transform"));
                }
                states[id] = 2;
            }
        }
        Ok(models)
    }
}

fn write_transform(value: &mut Value, transform: &Transform) {
    for (field, values) in [
        ("Translation", transform.translation.as_slice()),
        ("Rotation", transform.rotation.as_slice()),
        ("Scale", transform.scale.as_slice()),
    ] {
        let axes = if field == "Rotation" && value[field].get("i").is_some() {
            ["i", "j", "k", "r"]
        } else {
            ["X", "Y", "Z", "W"]
        };
        for (axis, component) in axes.into_iter().zip(values) {
            value[field][axis] = serde_json::json!(component);
        }
    }
}

fn parent_index(value: i64, count: usize) -> Result<Option<usize>, AssetError> {
    if value == -1 {
        return Ok(None);
    }
    let index = usize::try_from(value).map_err(|_| invalid("negative rig parent other than -1"))?;
    if index >= count {
        return Err(invalid("rig parent exceeds bone count"));
    }
    Ok(Some(index))
}

fn decode_appendix(bytes: &[u8], bones: &mut [Bone]) -> Result<(), AssetError> {
    if bytes.len()
        != bones
            .len()
            .checked_mul(50)
            .ok_or_else(|| invalid("rig appendix size overflow"))?
    {
        return Err(invalid("rig appendix size does not match bone count"));
    }
    let count = bones.len();
    for (index, bone) in bones.iter_mut().enumerate() {
        let offset = index * 2;
        let parent = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        bone.parent = parent_index(i64::from(parent), count)?;
        let offset = count * 2 + index * 48;
        let mut values = [0.0_f32; 12];
        for (component, value) in values.iter_mut().enumerate() {
            let offset = offset + component * 4;
            *value = f32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .map_err(|_| invalid("truncated rig transform"))?,
            );
        }
        bone.rest = Transform {
            translation: [values[0], values[1], values[2]],
            rotation: [values[4], values[5], values[6], values[7]],
            scale: [values[8], values[9], values[10]],
        };
        bone.rest
            .matrix()
            .map_err(|error| invalid(format!("rig bone {}: {error}", bone.name)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn geometry_a_pose_is_preserved_separately_from_animation_reference() {
        let mut document = json!({"$type":"animRig","boneNames":["Root"],"boneParentIndexes":[-1],
            "boneTransforms":[{"Translation":{"X":1.0},"Scale":{"X":1.0,"Y":1.0,"Z":1.0}}],
            "aPoseLS":[{"Translation":{"X":2.0},"Scale":{"X":1.0,"Y":1.0,"Z":1.0}}]});
        let rig = Rig::decode(&document).unwrap();
        assert!((rig.bones[0].rest.translation[0] - 1.).abs() < f32::EPSILON);
        assert!((rig.a_pose.as_ref().unwrap()[0].translation[0] - 2.).abs() < f32::EPSILON);
        Rig::scale_root_document(&mut document, [2., 1., 1.]).unwrap();
        assert!(
            (Rig::decode(&document).unwrap().a_pose.unwrap()[0].scale[0] - 2.).abs() < f32::EPSILON
        );
        document["aPoseLS"] = json!([{}, {}]);
        assert!(Rig::decode(&document).is_err());
        document["aPoseLS"] = json!([]);
        assert!(Rig::decode(&document).unwrap().a_pose.is_none());
    }

    #[test]
    fn small_authored_bone_scales_remain_invertible() {
        let document = json!({"$type":"animRig", "boneNames":["pos_ironsight"],
        "boneParentIndexes":[-1], "boneTransforms":[{
            "Scale":{"X":90.0,"Y":1e-12,"Z":180.0}
        }]});
        let rig = Rig::decode(&document).unwrap();
        let matrix = rig.model_rest().unwrap()[0];
        assert_eq!(matrix.y_axis.y, 1e-12);
        assert!(matrix.inverse().is_finite());
        let mut singular = document;
        singular["boneTransforms"][0]["Scale"]["Y"] = json!(0.0);
        assert!(Rig::decode(&singular).is_err());
    }

    #[test]
    fn scaled_native_root_preserves_child_transforms_and_other_appendix_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(-1_i16).to_le_bytes());
        bytes.extend_from_slice(&0_i16.to_le_bytes());
        for x in [0.0_f32, 2.0] {
            for value in [x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut document = json!({"$type":"animRig", "boneNames":["Root","gate"], "$rawTail":STANDARD.encode(&bytes)});
        Rig::scale_root_document(&mut document, [1.5, 1., 2.]).unwrap();
        let changed = STANDARD
            .decode(document["$rawTail"].as_str().unwrap())
            .unwrap();
        assert!(
            bytes
                .iter()
                .zip(&changed)
                .enumerate()
                .all(|(index, (a, b))| a == b || (36..48).contains(&index))
        );
        let rig = Rig::decode(&document).unwrap();
        assert_eq!(rig.bones[1].rest.translation, [2., 0., 0.]);
        assert_eq!(rig.model_rest().unwrap()[1].w_axis.x, 3.);
        let before = document.clone();
        assert!(Rig::scale_root_document(&mut document, [1., 0., 1.]).is_err());
        assert_eq!(document, before);
    }

    #[test]
    fn native_appendix_preserves_hierarchy_and_rejects_truncation_and_cycles() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_i16.to_le_bytes());
        bytes.extend_from_slice(&(-1_i16).to_le_bytes());
        for x in [2.0_f32, 3.0] {
            for value in [x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0.] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut document = json!({"$type":"animRig", "boneNames":["child","root"], "$rawTail":STANDARD.encode(&bytes)});
        let rig = Rig::decode(&document).unwrap();
        assert!((rig.model_rest().unwrap()[0].w_axis.x - 5.0).abs() < 1e-6);
        document["$rawTail"] = json!(STANDARD.encode(&bytes[..bytes.len() - 1]));
        assert!(Rig::decode(&document).is_err());
        bytes[2..4].copy_from_slice(&0_i16.to_le_bytes());
        document["$rawTail"] = json!(STANDARD.encode(&bytes));
        assert!(Rig::decode(&document).is_err());
    }
}
