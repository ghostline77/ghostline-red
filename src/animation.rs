//! Native keyframe and SIMD animation decoding and continuous pose sampling.

mod motion;
mod simd;
use motion::decode_motion;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use glam::{Quat, Vec3};
use serde_json::Value;

use crate::assets::{AssetError, AssetReader, ResourceKey, Transform, invalid, name, number, root};

/// Timestamped native animation value.
#[derive(Clone, Debug)]
pub struct Key<T> {
    /// Time in seconds from clip start.
    pub time: f32,
    /// Sample value.
    pub value: T,
}

/// Independent local transform channels for one bone.
#[derive(Clone, Debug, Default)]
pub struct JointChannels {
    /// Native Z-up translations.
    pub translation: Vec<Key<Vec3>>,
    /// Native XYZW quaternion rotations.
    pub rotation: Vec<Key<Quat>>,
    /// Native axis scales.
    pub scale: Vec<Key<Vec3>>,
}

/// Decoded animation clip, retaining its native rig identity and additive mode.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Native compressed-key or SIMD buffer type.
    pub buffer_type: String,
    /// Authored clip name.
    pub name: String,
    /// Native animation type, such as `Normal` or `AdditiveFromRefPose`.
    pub kind: String,
    /// Rig resource required by native joint indices.
    pub rig: Option<ResourceKey>,
    /// Clip duration in seconds.
    pub duration: f32,
    /// Authored frame count.
    pub frame_count: usize,
    /// Trailing synchronization joints, excluded by pose sampling.
    pub extra_joints: usize,
    /// Local bone channels in native index order.
    pub joints: Vec<JointChannels>,
    /// Scalar animation tracks, including facial/custom channels.
    pub tracks: Vec<Vec<Key<f32>>>,
    /// Authored motion-extraction data, retained for explicit root-motion handling.
    pub motion: Option<JointChannels>,
}

/// Continuous sample in native RED coordinates.
#[derive(Clone, Debug)]
pub struct AnimationSample {
    /// Local bone transforms, excluding synchronization-only joints.
    pub local: Vec<Transform>,
    /// Scalar channel values in native track order.
    pub tracks: Vec<f32>,
    /// Native motion-extraction transform sampled at the same timestamp.
    pub motion: Transform,
}

impl AnimationClip {
    /// Decodes a reflected animation buffer and its embedded key payload.
    ///
    /// # Errors
    /// Returns [`AssetError`] for unsupported formats, invalid dimensions or truncated keys.
    pub fn decode_buffer(buffer: &Value) -> Result<Self, AssetError> {
        let duration = number(&buffer["duration"])?;
        let frames = count(buffer, "numFrames", 0)?;
        let joints = count(buffer, "numJoints", 0)?;
        let extras = count(buffer, "numExtraJoints", 0)?;
        let tracks = count(buffer, "numTracks", 0)?;
        if duration < 0.0
            || frames == 0
            || frames > 60_000
            || joints == 0
            || joints > 8192
            || extras > joints
            || tracks > 65_535
        {
            return Err(invalid("invalid animation duration or dimensions"));
        }
        let payload = payload(buffer)?;
        let mut clip = Self {
            buffer_type: name(buffer, "$type"),
            name: String::new(),
            kind: "Normal".into(),
            rig: None,
            duration,
            frame_count: frames,
            extra_joints: extras,
            joints: vec![JointChannels::default(); joints],
            tracks: vec![Vec::new(); tracks],
            motion: None,
        };
        match buffer["$type"].as_str() {
            Some("animAnimationBufferCompressed") => {
                decode_compressed(&mut clip, buffer, &payload)?;
            }
            Some("animAnimationBufferSimd") => simd::decode(&mut clip, buffer, &payload)?,
            other => return Err(invalid(format!("unsupported animation buffer {other:?}"))),
        }
        for joint in &mut clip.joints {
            joint.translation.sort_by(|a, b| a.time.total_cmp(&b.time));
            joint.rotation.sort_by(|a, b| a.time.total_cmp(&b.time));
            joint.scale.sort_by(|a, b| a.time.total_cmp(&b.time));
        }
        for track in &mut clip.tracks {
            track.sort_by(|a, b| a.time.total_cmp(&b.time));
        }
        Ok(clip)
    }

    /// Samples clamped clip time using linear translation/scale and quaternion slerp.
    ///
    /// Missing normal channels use `reference`. Additive clips require an explicit
    /// base pose, onto which native local deltas are composed. Synchronization-only
    /// joints are excluded; the reference and base must match the body joint count.
    ///
    /// # Errors
    /// Returns [`AssetError`] for nonfinite time, rig-count mismatch or missing additive base.
    pub fn sample(
        &self,
        time: f32,
        reference: &[Transform],
        additive_base: Option<&[Transform]>,
    ) -> Result<AnimationSample, AssetError> {
        if !time.is_finite() {
            return Err(invalid("nonfinite animation sample time"));
        }
        let count = self.joints.len() - self.extra_joints;
        if reference.len() != count || additive_base.is_some_and(|b| b.len() != count) {
            return Err(invalid("animation pose count differs from source rig"));
        }
        let additive = matches!(
            self.kind.as_str(),
            "Additive"
                | "AdditiveFromRefPose"
                | "AdditiveFromFirstFrame"
                | "AdditiveWithoutFirstFrame"
        );
        if additive && additive_base.is_none() {
            return Err(invalid(format!(
                "{} requires an explicit additive base pose",
                self.kind
            )));
        }
        if !additive && self.kind != "Normal" && self.kind != "NormalWithTrajectory" {
            return Err(invalid(format!("unsupported animation mode {}", self.kind)));
        }
        let time = time.clamp(0., self.duration);
        let mut local = Vec::with_capacity(count);
        for (index, channels) in self.joints[..count].iter().enumerate() {
            let defaults = if additive {
                Transform::default()
            } else {
                reference[index]
            };
            let mut transform = sample_joint(channels, time, defaults);
            if let Some(base) = additive_base.filter(|_| additive) {
                transform.translation = (Vec3::from(base[index].translation)
                    + Vec3::from(transform.translation))
                .to_array();
                transform.rotation = (Quat::from_array(base[index].rotation).normalize()
                    * Quat::from_array(transform.rotation))
                .normalize()
                .to_array();
                transform.scale =
                    (Vec3::from(base[index].scale) * Vec3::from(transform.scale)).to_array();
            }
            transform.matrix()?;
            local.push(transform);
        }
        let tracks = self
            .tracks
            .iter()
            .map(|keys| sample_keys(keys, time, 0., |a, b, t| a + (b - a) * t))
            .collect();
        let motion = self
            .motion
            .as_ref()
            .map_or_else(Transform::default, |channels| {
                sample_joint(channels, time, Transform::default())
            });
        Ok(AnimationSample {
            local,
            tracks,
            motion,
        })
    }
}

impl AssetReader<'_> {
    /// Loads an exact named clip from a native animation set, including streamed data.
    ///
    /// # Errors
    /// Returns [`AssetError`] for absent/ambiguous names, invalid handles or unsupported data.
    pub fn animation(
        &mut self,
        resource: &ResourceKey,
        requested: &str,
    ) -> Result<AnimationClip, AssetError> {
        let document = self.document(resource)?;
        let root = root(&document);
        let handles = super::assets::assembly::handles(root)?;
        let entries = root["animations"]
            .as_array()
            .ok_or_else(|| invalid("animation set has no entries"))?;
        let mut selected = None;
        for entry in entries {
            let entry = super::assets::assembly::resolve_handle(entry, &handles)?;
            let animation = super::assets::assembly::resolve_handle(&entry["animation"], &handles)?;
            if name(animation, "name") == requested {
                if selected.is_some() {
                    return Err(invalid(format!("ambiguous animation {requested}")));
                }
                selected = Some(animation);
            }
        }
        let animation =
            selected.ok_or_else(|| invalid(format!("animation {requested:?} is absent")))?;
        let mut buffer =
            super::assets::assembly::resolve_handle(&animation["animBuffer"], &handles)?.clone();
        if !["defferedBuffer", "inplaceCompressedBuffer", "tempBuffer"]
            .iter()
            .any(|field| buffer.get(field).is_some_and(|b| b.get("Bytes").is_some()))
        {
            let address = &buffer["dataAddress"];
            let index = count(address, "unkIndex", usize::MAX)?;
            let offset = count(address, "fsetInBytes", usize::MAX)?;
            let size = count(address, "zeInBytes", usize::MAX)?;
            let encoded = root["animationDataChunks"][index]["buffer"]["Bytes"]
                .as_str()
                .ok_or_else(|| invalid("animation stream chunk is absent"))?;
            let bytes = STANDARD.decode(encoded)?;
            let bytes = bytes
                .get(
                    offset
                        ..offset
                            .checked_add(size)
                            .ok_or_else(|| invalid("animation address overflow"))?,
                )
                .ok_or_else(|| invalid("animation stream address exceeds chunk"))?;
            buffer["defferedBuffer"] = serde_json::json!({"Bytes":STANDARD.encode(bytes)});
        }
        let mut clip = AnimationClip::decode_buffer(&buffer)?;
        requested.clone_into(&mut clip.name);
        animation
            .get("animationType")
            .and_then(Value::as_str)
            .unwrap_or("Normal")
            .clone_into(&mut clip.kind);
        clip.rig = ResourceKey::from_value(&root["rig"])?;
        let motion =
            super::assets::assembly::resolve_handle(&animation["motionExtraction"], &handles)?;
        clip.motion = decode_motion(motion, clip.duration)?;
        Ok(clip)
    }
}

fn sample_joint(channels: &JointChannels, time: f32, defaults: Transform) -> Transform {
    Transform {
        translation: sample_keys(
            &channels.translation,
            time,
            defaults.translation.into(),
            |a, b, t| a.lerp(*b, t),
        )
        .to_array(),
        rotation: sample_keys(
            &channels.rotation,
            time,
            Quat::from_array(defaults.rotation),
            |a, b, t| a.slerp(*b, t).normalize(),
        )
        .to_array(),
        scale: sample_keys(&channels.scale, time, defaults.scale.into(), |a, b, t| {
            a.lerp(*b, t)
        })
        .to_array(),
    }
}

fn sample_keys<T: Copy>(
    keys: &[Key<T>],
    time: f32,
    default: T,
    lerp: impl Fn(&T, &T, f32) -> T,
) -> T {
    let index = keys.partition_point(|key| key.time <= time);
    if index == 0 {
        return keys.first().map_or(default, |k| k.value);
    }
    if index == keys.len() {
        return keys[index - 1].value;
    }
    let (left, right) = (&keys[index - 1], &keys[index]);
    let span = right.time - left.time;
    lerp(
        &left.value,
        &right.value,
        if span > 0. {
            (time - left.time) / span
        } else {
            1.
        },
    )
}

fn payload(buffer: &Value) -> Result<Vec<u8>, AssetError> {
    for field in ["inplaceCompressedBuffer", "defferedBuffer", "tempBuffer"] {
        if let Some(bytes) = buffer[field]["Bytes"].as_str() {
            return Ok(STANDARD.decode(bytes)?);
        }
    }
    Err(invalid(
        "animation has no embedded or resolved streamed payload",
    ))
}

fn decode_compressed(
    clip: &mut AnimationClip,
    buffer: &Value,
    bytes: &[u8],
) -> Result<(), AssetError> {
    let mut cursor = Cursor::new(bytes);
    for (field, quantized, constant) in [
        ("numAnimKeys", true, false),
        ("numAnimKeysRaw", false, false),
        ("numConstAnimKeys", false, true),
    ] {
        let count = count(buffer, field, 0)?;
        if count > bytes.len() / if quantized { 10 } else { 16 } {
            return Err(invalid("animation key count exceeds payload"));
        }
        for _ in 0..count {
            let a = cursor.u16()?;
            let b = cursor.u16()?;
            let (time, packed) = if constant { (b, a) } else { (a, b) };
            let mut value = [0.0; 3];
            for component in &mut value {
                *component = if quantized {
                    f32::from(cursor.u16()?) / 65535. * 2. - 1.
                } else {
                    cursor.f32()?
                };
            }
            let joint = usize::from(packed & 0x1fff);
            let channel = (packed >> 13) & 3;
            let time = f32::from(time) / 65535. * clip.duration;
            let channels = clip
                .joints
                .get_mut(joint)
                .ok_or_else(|| invalid("animation key joint exceeds rig"))?;
            match channel {
                0 => channels.translation.push(Key {
                    time,
                    value: value.into(),
                }),
                1 => channels.rotation.push(Key {
                    time,
                    value: packed_rotation(value.into(), packed & 0x8000 != 0)?,
                }),
                2 => channels.scale.push(Key {
                    time,
                    value: value.into(),
                }),
                _ => return Err(invalid("invalid animation key component")),
            }
        }
    }
    for (field, constant) in [("numTrackKeys", false), ("numConstTrackKeys", true)] {
        let count = count(buffer, field, 0)?;
        if count > bytes.len() / 8 {
            return Err(invalid("animation track count exceeds payload"));
        }
        for _ in 0..count {
            let a = cursor.u16()?;
            let b = cursor.u16()?;
            let (time, index) = if constant { (b, a) } else { (a, b) };
            let value = cursor.f32()?;
            clip.tracks
                .get_mut(usize::from(index))
                .ok_or_else(|| invalid("animation track index exceeds count"))?
                .push(Key {
                    time: f32::from(time) / 65535. * clip.duration,
                    value,
                });
        }
    }
    Ok(())
}

fn packed_rotation(value: Vec3, negative: bool) -> Result<Quat, AssetError> {
    let dot = value.length_squared();
    let xyz = value * (2. - dot).max(0.).sqrt();
    let w = (1. - dot) * if negative { -1. } else { 1. };
    let q = Quat::from_xyzw(xyz.x, xyz.y, xyz.z, w);
    if !q.is_finite() || q.length_squared() < 1e-12 {
        return Err(invalid("invalid animation quaternion"));
    }
    Ok(q.normalize())
}

fn count(value: &Value, field: &str, default: usize) -> Result<usize, AssetError> {
    value.get(field).map_or(Ok(default), |v| {
        v.as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid(format!("invalid animation count {field}")))
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take<const N: usize>(&mut self) -> Result<[u8; N], AssetError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or_else(|| invalid("animation buffer offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("truncated animation payload"))?;
        self.offset = end;
        bytes
            .try_into()
            .map_err(|_| invalid("invalid animation element width"))
    }
    fn u16(&mut self) -> Result<u16, AssetError> {
        Ok(u16::from_le_bytes(self.take()?))
    }
    fn f32(&mut self) -> Result<f32, AssetError> {
        let value = f32::from_le_bytes(self.take()?);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid("nonfinite animation value"))
        }
    }
    fn floats(&mut self, count: usize) -> Result<Vec<f32>, AssetError> {
        if count > self.bytes.len().saturating_sub(self.offset) / 4 {
            return Err(invalid("truncated animation float stream"));
        }
        (0..count).map(|_| self.f32()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn raw_key(bytes: &mut Vec<u8>, time: u16, packed: u16, value: [f32; 3]) {
        bytes.extend_from_slice(&time.to_le_bytes());
        bytes.extend_from_slice(&packed.to_le_bytes());
        for component in value {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }

    #[test]
    fn compressed_sampling_interpolates_channels_tracks_and_requires_additive_base() {
        let mut bytes = Vec::new();
        raw_key(&mut bytes, 0, 0, [0.; 3]);
        raw_key(&mut bytes, u16::MAX, 0, [2., 0., 0.]);
        raw_key(&mut bytes, 0, 0x2000, [0.; 3]);
        raw_key(&mut bytes, u16::MAX, 0x2000, [0., 0., 1.]);
        // Constant keys store the joint/component before the time field.
        raw_key(&mut bytes, 1, 0, [0., 1., 0.]);
        for (time, value) in [(0_u16, 0_f32), (u16::MAX, 1.)] {
            bytes.extend_from_slice(&time.to_le_bytes());
            bytes.extend_from_slice(&0_u16.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut buffer = json!({"$type":"animAnimationBufferCompressed","duration":2.,"numFrames":3,"numJoints":2,"numTracks":1,"numAnimKeysRaw":4,"numConstAnimKeys":1,"numTrackKeys":2,"defferedBuffer":{"Bytes":STANDARD.encode(&bytes)}});
        let mut clip = AnimationClip::decode_buffer(&buffer).unwrap();
        let reference = [Transform::default(); 2];
        let sample = clip.sample(1., &reference, None).unwrap();
        assert!((sample.local[0].translation[0] - 1.).abs() < 1e-6);
        assert!((sample.local[1].translation[1] - 1.).abs() < 1e-6);
        assert!((sample.local[0].rotation[2].abs() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((sample.tracks[0] - 0.5).abs() < 1e-6);
        clip.kind = "AdditiveFromRefPose".into();
        assert!(clip.sample(1., &reference, None).is_err());
        let mut base = reference;
        base[0].translation[0] = 3.;
        assert!(
            (clip.sample(1., &reference, Some(&base)).unwrap().local[0].translation[0] - 4.).abs()
                < 1e-6
        );
        buffer["defferedBuffer"]["Bytes"] = json!(STANDARD.encode(&bytes[..bytes.len() - 1]));
        assert!(AnimationClip::decode_buffer(&buffer).is_err());
    }

    #[test]
    fn simd_sampling_preserves_copy_translations_padding_and_constant_tracks() {
        let mut bytes = Vec::new();
        let mut floats = |values: &[f32]| {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        };
        for _ in 0..2 {
            floats(&[
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
            ]);
        }
        floats(&[0.; 12]);
        floats(&[2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
        floats(&[1., 1., 1., 0.]);
        floats(&[0.25]);
        floats(&[0., 2., 0.]);
        for index in [2_u16, 0, 1, u16::MAX, u16::MAX] {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        let buffer = json!({"$type":"animAnimationBufferSimd","duration":2.,"numFrames":2,"numJoints":3,"numTracks":1,"numTranslationsToEvalAlignedToSimd":4,"numTranslationsToCopy":1,"isScaleConstant":true,"isTrackConstant":true,"defferedBuffer":{"Bytes":STANDARD.encode(bytes)}});
        let clip = AnimationClip::decode_buffer(&buffer).unwrap();
        let sample = clip.sample(1., &[Transform::default(); 3], None).unwrap();
        assert!((sample.local[0].translation[0] - 1.).abs() < 1e-6);
        assert!((sample.local[2].translation[1] - 2.).abs() < 1e-6);
        assert!((sample.tracks[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn quantized_simd_rotation_reads_values_across_byte_boundaries() {
        let bits = 15;
        let values = 24;
        let mask = (1_u32 << bits) - 1;
        let zero = mask.div_ceil(2);
        let mut bytes = vec![0_u8; 48];
        for value in 0..values {
            for bit in 0..bits {
                if zero & (1 << bit) != 0 {
                    let position = value * bits + bit;
                    bytes[position / 8] |= 1 << (position % 8);
                }
            }
        }
        for value in [1_f32, 1., 1., 0.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = json!({"$type":"animAnimationBufferSimd","duration":1.,"numFrames":2,"numJoints":1,"numTracks":0,"quantizationBits":bits,"isScaleConstant":true,"defferedBuffer":{"Bytes":STANDARD.encode(bytes)}});
        let clip = AnimationClip::decode_buffer(&buffer).unwrap();
        let sample = clip.sample(0.5, &[Transform::default()], None).unwrap();
        assert!(Quat::from_array(sample.local[0].rotation).abs_diff_eq(Quat::IDENTITY, 1e-3));
    }
}
