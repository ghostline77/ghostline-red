use super::{
    AssetError, Cursor, JointChannels, Key, Quat, Transform, Value, Vec3, invalid, number,
    packed_rotation,
};

pub(super) fn decode_motion(
    value: &Value,
    clip_duration: f32,
) -> Result<Option<JointChannels>, AssetError> {
    if value.is_null() || value.get("$type").is_none() {
        return Ok(None);
    }
    let duration = value
        .get("duration")
        .map(number)
        .transpose()?
        .unwrap_or(clip_duration);
    if duration < 0.0 {
        return Err(invalid("negative root-motion duration"));
    }
    let mut result = JointChannels::default();
    match value["$type"].as_str() {
        Some("animLinearCompressedMotionExtraction") => {
            decode_linear(value, &mut result)?;
        }
        Some(
            "animUncompressedMotionExtraction"
            | "animPlaneUncompressedMotionExtraction"
            | "animUncompressedAllAnglesMotionExtraction",
        ) => {
            decode_uniform(value, &mut result, duration)?;
        }
        Some("animSplineCompressedMotionExtraction") => {
            decode_spline(value, &mut result, duration)?;
        }
        other => {
            return Err(invalid(format!(
                "unsupported native motion extraction {other:?}"
            )));
        }
    }
    result.translation.sort_by(|a, b| a.time.total_cmp(&b.time));
    result.rotation.sort_by(|a, b| a.time.total_cmp(&b.time));
    if result
        .translation
        .iter()
        .any(|k| k.time < 0. || k.time > duration + 1e-3)
        || result
            .rotation
            .iter()
            .any(|k| k.time < 0. || k.time > duration + 1e-3)
    {
        return Err(invalid("root-motion key time exceeds duration"));
    }
    Ok(Some(result))
}

fn array(value: &Value) -> Result<&[Value], AssetError> {
    if value.is_null() {
        return Ok(&[]);
    }
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("invalid motion-extraction array"))
}
fn vec(value: &Value) -> Result<Vec3, AssetError> {
    Ok(Vec3::new(
        number(&value["X"])?,
        number(&value["Y"])?,
        number(&value["Z"])?,
    ))
}

fn decode_linear(value: &Value, result: &mut JointChannels) -> Result<(), AssetError> {
    let positions = array(&value["posFrames"])?;
    let rotations = array(&value["rotFrames"])?;
    let pos_times = array(&value["posTime"])?;
    let rot_times = array(&value["rotTime"])?;
    if positions.len() != pos_times.len() || rotations.len() != rot_times.len() {
        return Err(invalid("root-motion value/time counts differ"));
    }
    for (position, time) in positions.iter().zip(pos_times) {
        result.translation.push(Key {
            time: number(time)?,
            value: vec(position)?,
        });
    }
    for (rotation, time) in rotations.iter().zip(rot_times) {
        let transform = Transform::decode(&serde_json::json!({"Rotation":rotation}))?;
        result.rotation.push(Key {
            time: number(time)?,
            value: Quat::from_array(transform.rotation).normalize(),
        });
    }

    Ok(())
}

fn decode_uniform(
    value: &Value,
    result: &mut JointChannels,
    duration: f32,
) -> Result<(), AssetError> {
    let frames = array(&value["frames"])?;
    for (index, frame) in frames.iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "native frame arrays are memory-bounded"
        )]
        let time = if frames.len() <= 1 {
            0.
        } else {
            duration * index as f32 / (frames.len() - 1) as f32
        };
        let transform = match value["$type"].as_str() {
            Some("animUncompressedAllAnglesMotionExtraction") => Transform::decode(frame)?,
            Some("animPlaneUncompressedMotionExtraction") => Transform {
                translation: [number(&frame["X"])?, number(&frame["Y"])?, 0.],
                rotation: Quat::from_rotation_z(number(&frame["Z"])?.to_radians()).to_array(),
                ..Transform::default()
            },
            _ => Transform {
                translation: vec(frame)?.to_array(),
                rotation: Quat::from_rotation_z(number(&frame["W"])?.to_radians()).to_array(),
                ..Transform::default()
            },
        };
        result.translation.push(Key {
            time,
            value: transform.translation.into(),
        });
        result.rotation.push(Key {
            time,
            value: Quat::from_array(transform.rotation).normalize(),
        });
    }

    Ok(())
}

fn decode_spline(
    value: &Value,
    result: &mut JointChannels,
    duration: f32,
) -> Result<(), AssetError> {
    for (field, component) in [("posKeysData", 0_u16), ("rotKeysData", 1_u16)] {
        let bytes = array(&value[field])?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or_else(|| invalid("invalid spline root-motion byte"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if bytes.len() % 16 != 0 {
            return Err(invalid("invalid spline root-motion key size"));
        }
        let mut cursor = Cursor::new(&bytes);
        for _ in 0..bytes.len() / 16 {
            let time = f32::from(cursor.u16()?) / 65535. * duration;
            let packed = cursor.u16()?;
            if packed & 0x1fff != 0 || (packed >> 13) & 3 != component {
                return Err(invalid(
                    "spline motion key targets an invalid joint/component",
                ));
            }
            let value = Vec3::new(cursor.f32()?, cursor.f32()?, cursor.f32()?);
            if component == 0 {
                result.translation.push(Key { time, value });
            } else {
                result.rotation.push(Key {
                    time,
                    value: packed_rotation(value, packed & 0x8000 != 0)?,
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn motion_extractors_preserve_degree_yaw_and_normalized_key_times() {
        let motion = decode_motion(&json!({"$type":"animUncompressedMotionExtraction","duration":2.,"frames":[{"X":0.,"Y":0.,"Z":0.,"W":0.},{"X":2.,"Y":3.,"Z":1.,"W":90.}]}),2.).unwrap().unwrap();
        assert!(
            motion.translation[1]
                .value
                .abs_diff_eq(Vec3::new(2., 3., 1.), 1e-6)
        );
        assert!(
            motion.rotation[1]
                .value
                .abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), 1e-6)
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u16::MAX.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        for value in [2_f32, 0., 0.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let motion = decode_motion(&json!({"$type":"animSplineCompressedMotionExtraction","duration":2.,"posKeysData":bytes,"rotKeysData":[]}),2.).unwrap().unwrap();
        assert!((motion.translation[0].time - 2.).abs() < 1e-6);
        assert!(motion.translation[0].value.abs_diff_eq(Vec3::X * 2., 1e-6));
        assert!(decode_motion(&json!({"$type":"unknownMotion"}), 2.).is_err());
    }
}
