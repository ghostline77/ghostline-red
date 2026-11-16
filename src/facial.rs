//! Native additive facial animation sets generated from named scalar curves.
use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

/// Builds an animation set without cloning skeletal keys, clips or resource documents.
/// Track names are resolved against the supplied native rig's exact track order.
///
/// # Errors
/// Rejects missing tracks, duplicate names, invalid dimensions and non-finite samples.
pub fn animation_set(rig_path: &str, rig: &Value, clips: &[Value]) -> Result<Value> {
    let root = rig.pointer("/Data/RootChunk").unwrap_or(rig);
    let names = root["trackNames"]
        .as_array()
        .context("facial rig trackNames")?;
    let track_name = |v: &Value| {
        v.as_str()
            .or_else(|| v["$value"].as_str())
            .map(str::to_owned)
            .context("rig track name")
    };
    let names = names.iter().map(track_name).collect::<Result<Vec<_>>>()?;
    let indices = names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        indices.len() == names.len() && names.len() <= usize::from(u16::MAX),
        "invalid facial track table"
    );
    let joints = root["boneNames"].as_array().context("rig boneNames")?.len();
    ensure!(joints > 0 && joints <= 8192, "invalid facial joint count");
    ensure!(!clips.is_empty(), "no facial clips");
    let mut animations = Vec::new();
    let mut chunks = Vec::new();
    let mut seen = HashSet::new();
    for (clip_index, clip) in clips.iter().enumerate() {
        let name = clip["name"].as_str().context("clip name")?;
        ensure!(seen.insert(name), "duplicate facial clip {name}");
        let duration = clip["duration"].as_f64().context("clip duration")?;
        let times = clip["times"].as_array().context("clip times")?;
        let tracks = clip["tracks"].as_array().context("clip tracks")?;
        let values = clip["values"].as_array().context("clip values")?;
        ensure!(
            duration.is_finite()
                && duration > 0.
                && times.len() >= 2
                && times.len() <= 60_000
                && values.len() == times.len(),
            "invalid facial dimensions"
        );
        let mut destinations = BTreeMap::new();
        for (column, track) in tracks.iter().enumerate() {
            let track = track.as_str().context("curve name")?;
            let index = *indices
                .get(track)
                .with_context(|| format!("rig lacks facial track {track}"))?;
            ensure!(
                destinations.insert(index, (column, 1_f32)).is_none(),
                "duplicate curve track {track}"
            );
            if let Some(stem) = track.strip_suffix("LipsyncPoseOutput") {
                if let Some(index) = indices.get(format!("{stem}AnimOverrideWeight").as_str()) {
                    ensure!(
                        destinations.insert(*index, (column, -1_f32)).is_none(),
                        "duplicate facial override"
                    );
                }
            }
        }
        let mut payload = Vec::new();
        // A native lipsync clip still supplies neutral additive skeletal
        // channels. Keep a two-key root marker so the engine has a full-duration
        // pose stream even though all authored movement is in scalar tracks.
        for time in [0_u16, u16::MAX] {
            payload.extend(time.to_le_bytes());
            payload.extend(0_u16.to_le_bytes());
            for _ in 0..3 {
                payload.extend(0_f32.to_le_bytes());
            }
        }
        for joint in 0..joints {
            for channel in 0..3_u16 {
                if joint == 0 && channel == 0 {
                    continue;
                }
                payload.extend((u16::try_from(joint)? | (channel << 13)).to_le_bytes());
                payload.extend(0_u16.to_le_bytes());
                let value = if channel == 2 { 1_f32 } else { 0_f32 };
                for _ in 0..3 {
                    payload.extend(value.to_le_bytes());
                }
            }
        }
        let scalar_offset = payload.len();
        let mut previous = -1.;
        let mut encoded_times = Vec::with_capacity(times.len());
        for (frame, time) in times.iter().enumerate() {
            let time = time.as_f64().context("facial time")?;
            ensure!(
                time.is_finite() && time >= 0. && time > previous && time <= duration + 1e-6,
                "invalid facial time sequence"
            );
            previous = time;
            let time = ((time / duration).clamp(0., 1.) * 65535.).round() as u16;
            let row = values[frame].as_array().context("facial frame")?;
            ensure!(row.len() == tracks.len(), "facial frame width");
            encoded_times.push(time);
        }
        // RED's scalar curves are contiguous runs ordered by track, then time.
        // Our decoder buckets keys by index and would hide an interleaved layout.
        for (track, (column, sign)) in &destinations {
            for (frame, time) in encoded_times.iter().enumerate() {
                let row = values[frame].as_array().context("facial frame")?;
                let value = row[*column].as_f64().context("facial scalar")? as f32 * sign;
                ensure!(value.is_finite(), "non-finite facial scalar");
                payload.extend(time.to_le_bytes());
                payload.extend(u16::try_from(*track)?.to_le_bytes());
                payload.extend(value.to_le_bytes());
            }
        }
        let dynamic_count = (payload.len() - scalar_offset) / 8;
        for track in 0..names.len() {
            if !destinations.contains_key(&track) {
                payload.extend(u16::try_from(track)?.to_le_bytes());
                payload.extend(0_u16.to_le_bytes());
                payload.extend(0_f32.to_le_bytes());
            }
        }
        let mut buffer = json!({"$type":"animAnimationBufferCompressed","duration":duration,"numFrames":times.len(),"numJoints":joints,"numTracks":names.len(),"numAnimKeys":0,"numAnimKeysRaw":2,"numConstAnimKeys":joints*3-1,"numTrackKeys":dynamic_count,"numConstTrackKeys":names.len()-destinations.len(),"isScaleConstant":true,"dataAddress":{"$type":"animAnimDataAddress","unkIndex":clip_index,"fsetInBytes":0,"zeInBytes":payload.len()}});
        buffer["defferedBuffer"] = json!({"Bytes":STANDARD.encode(&payload)});
        crate::animation::AnimationClip::decode_buffer(&buffer)?;
        buffer
            .as_object_mut()
            .context("animation buffer")?
            .remove("defferedBuffer");
        let handle = clip_index * 3;
        animations.push(json!({"HandleId":handle.to_string(),"Data":{"$type":"animAnimSetEntry","animation":{"HandleId":(handle+1).to_string(),"Data":{"$type":"animAnimation","name":{"$type":"CName","$storage":"string","$value":name},"duration":duration,"animationType":"AdditiveFromRefPose","animBuffer":{"HandleId":(handle+2).to_string(),"Data":buffer},"motionExtraction":null}},"events":null}}));
        chunks.push(json!({"$type":"animAnimDataChunk","buffer":{"BufferId":clip_index.to_string(),"Flags":0,"Bytes":STANDARD.encode(payload)}}));
    }
    Ok(
        json!({"Header":{"WKitJsonVersion":"0.0.9","GameVersion":2000,"ExportedDateTime":"1970-01-01T00:00:00Z","DataType":"CR2W"},"Data":{"Version":195,"BuildVersion":0,"RootChunk":{"$type":"animAnimSet","cookingPlatform":"PLATFORM_PC","rig":{"DepotPath":{"$type":"ResourcePath","$storage":"string","$value":rig_path},"Flags":"Default"},"animations":animations,"animationDataChunks":chunks},"EmbeddedFiles":[]}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facial_payload_groups_native_track_runs_and_preserves_constants() {
        let rig = json!({"boneNames":["root"],"trackNames":["aLipsyncPoseOutput","aAnimOverrideWeight","idle"]});
        let clip = json!({"name":"speech","duration":1.,"times":[0.,1.],"tracks":["aLipsyncPoseOutput"],"values":[[0.25],[0.75]]});
        let document = animation_set("base\\face.rig", &rig, &[clip]).unwrap();
        let root = &document["Data"]["RootChunk"];
        let bytes = STANDARD
            .decode(
                root["animationDataChunks"][0]["buffer"]["Bytes"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
        let scalar = &bytes[64..];
        let indices = scalar[..32]
            .chunks_exact(8)
            .map(|key| u16::from_le_bytes([key[2], key[3]]))
            .collect::<Vec<_>>();
        assert_eq!(indices, [0, 0, 1, 1]);
        assert_eq!(u16::from_le_bytes([scalar[34], scalar[35]]), 0);
        assert_eq!(
            f32::from_le_bytes(scalar[28..32].try_into().unwrap()),
            -0.75
        );
        let animation = &root["animations"][0]["Data"]["animation"]["Data"]["animBuffer"]["Data"];
        assert_eq!(animation["numAnimKeysRaw"], 2);
        assert_eq!(animation["numConstAnimKeys"], 2);
    }
}
