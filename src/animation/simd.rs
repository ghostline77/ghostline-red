use super::{
    AnimationClip, AssetError, Cursor, JointChannels, Key, Quat, Value, Vec3, count, invalid,
    packed_rotation,
};

pub(super) fn decode(
    clip: &mut AnimationClip,
    buffer: &Value,
    bytes: &[u8],
) -> Result<(), AssetError> {
    let frames = clip.frame_count;
    let body = clip.joints.len() - clip.extra_joints;
    let aligned = (body + 3) & !3;
    let values = frames
        .checked_mul(aligned)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| invalid("SIMD sample count overflow"))?;
    if frames < 2 || values > 12_000_000 {
        return Err(invalid("SIMD animation exceeds sample budget"));
    }
    let bits = count(buffer, "quantizationBits", 0)?;
    let eval_count = count(buffer, "numTranslationsToEvalAlignedToSimd", 0)?;
    let copy_count = count(buffer, "numTranslationsToCopy", 0)?;
    if eval_count > aligned || eval_count % 4 != 0 || copy_count > clip.joints.len() {
        return Err(invalid("invalid SIMD translation dimensions"));
    }
    let mut cursor = Cursor::new(bytes);
    decode_rotations(clip, &mut cursor, aligned, bits)?;
    let translations = cursor.floats(frames * eval_count * 3)?;
    if flag(buffer, "isScaleConstant")? {
        let values = cursor.floats(4)?;
        let value = Vec3::new(values[0], values[1], values[2]);
        for joint in &mut clip.joints[..body] {
            joint.scale.push(Key { time: 0., value });
        }
    } else {
        let values = cursor.floats(frames * aligned * 3)?;
        for frame in 0..frames {
            let time = frame_time(clip, frame);
            for block in (0..aligned).step_by(4) {
                let base = frame * aligned * 3 + block * 3;
                for lane in 0..4 {
                    let joint = block + lane;
                    if joint < body {
                        clip.joints[joint].scale.push(Key {
                            time,
                            value: Vec3::new(
                                values[base + lane],
                                values[base + 4 + lane],
                                values[base + 8 + lane],
                            ),
                        });
                    }
                }
            }
        }
    }
    decode_tracks(clip, buffer, &mut cursor)?;
    let copy_positions = cursor.floats(copy_count * 3)?;
    let copy_indices = (0..copy_count)
        .map(|_| cursor.u16())
        .collect::<Result<Vec<_>, _>>()?;
    let eval_indices = (0..eval_count)
        .map(|_| cursor.u16())
        .collect::<Result<Vec<_>, _>>()?;
    for frame in 0..frames {
        let time = frame_time(clip, frame);
        for block in (0..eval_count).step_by(4) {
            let base = frame * eval_count * 3 + block * 3;
            for lane in 0..4 {
                if let Some(joint) = joint_mut(&mut clip.joints, eval_indices[block + lane])? {
                    joint.translation.push(Key {
                        time,
                        value: Vec3::new(
                            translations[base + lane],
                            translations[base + 4 + lane],
                            translations[base + 8 + lane],
                        ),
                    });
                }
            }
        }
    }
    for (index, &bone) in copy_indices.iter().enumerate() {
        if let Some(joint) = joint_mut(&mut clip.joints, bone)? {
            let base = index * 3;
            joint.translation.clear();
            joint.translation.push(Key {
                time: 0.,
                value: Vec3::new(
                    copy_positions[base],
                    copy_positions[base + 1],
                    copy_positions[base + 2],
                ),
            });
        }
    }
    Ok(())
}

fn joint_mut(
    joints: &mut [JointChannels],
    index: u16,
) -> Result<Option<&mut JointChannels>, AssetError> {
    if index == u16::MAX {
        return Ok(None);
    }
    joints
        .get_mut(usize::from(index))
        .map(Some)
        .ok_or_else(|| invalid("SIMD translation index exceeds joint count"))
}

fn flag(buffer: &Value, name: &str) -> Result<bool, AssetError> {
    buffer.get(name).map_or(Ok(false), |v| {
        v.as_bool()
            .or_else(|| v.as_u64().map(|n| n != 0))
            .ok_or_else(|| invalid("invalid SIMD constant flag"))
    })
}

fn frame_time(clip: &AnimationClip, frame: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "animation frame counts are bounded below 60001"
    )]
    {
        clip.duration * frame as f32 / (clip.frame_count - 1) as f32
    }
}

fn decode_rotations(
    clip: &mut AnimationClip,
    cursor: &mut Cursor<'_>,
    aligned: usize,
    bits: usize,
) -> Result<(), AssetError> {
    let frames = clip.frame_count;
    let body = clip.joints.len() - clip.extra_joints;
    let values = frames * aligned * 3;
    if bits != 0 {
        if bits > 16 {
            return Err(invalid("SIMD rotation quantization exceeds 16 bits"));
        }
        let packed_values = (values + 3) & !3;
        let compressed = (packed_values * bits).div_ceil(8).next_multiple_of(16);
        if compressed > cursor.bytes.len() {
            return Err(invalid("truncated SIMD rotation stream"));
        }
        let mask = (1_u32 << bits) - 1;
        let mut unpacked = Vec::with_capacity(packed_values);
        for index in 0..packed_values {
            let bit = index * bits;
            let offset = bit / 8;
            let mut packed = [0_u8; 4];
            let available = (compressed - offset).min(4);
            packed[..available].copy_from_slice(&cursor.bytes[offset..offset + available]);
            #[expect(
                clippy::cast_precision_loss,
                reason = "quantization uses at most sixteen bits"
            )]
            let value =
                ((u32::from_le_bytes(packed) >> (bit % 8)) & mask) as f32 / mask as f32 * 2. - 1.;
            unpacked.push(value);
        }
        for frame in 0..frames {
            let time = frame_time(clip, frame);
            for block in (0..aligned).step_by(4) {
                let base = frame * aligned * 3 + block * 3;
                for lane in 0..4 {
                    let joint = block + lane;
                    if joint >= body {
                        continue;
                    }
                    let value = packed_rotation(
                        Vec3::new(
                            unpacked[base + lane],
                            unpacked[base + 4 + lane],
                            unpacked[base + 8 + lane],
                        ),
                        false,
                    )?;
                    clip.joints[joint].rotation.push(Key { time, value });
                }
            }
        }
        cursor.offset = compressed;
    } else {
        for frame in 0..frames {
            let time = frame_time(clip, frame);
            for block in (0..aligned).step_by(4) {
                let values = cursor.floats(16)?;
                for lane in 0..4 {
                    let joint = block + lane;
                    if joint >= body {
                        continue;
                    }
                    let value = Quat::from_xyzw(
                        values[lane],
                        values[4 + lane],
                        values[8 + lane],
                        values[12 + lane],
                    );
                    if value.length_squared() < 1e-12 {
                        return Err(invalid("zero SIMD rotation quaternion"));
                    }
                    clip.joints[joint].rotation.push(Key {
                        time,
                        value: value.normalize(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn decode_tracks(
    clip: &mut AnimationClip,
    buffer: &Value,
    cursor: &mut Cursor<'_>,
) -> Result<(), AssetError> {
    let frames = clip.frame_count;
    if !clip.tracks.is_empty() {
        if flag(buffer, "isTrackConstant")? {
            let value = cursor.f32()?;
            for track in &mut clip.tracks {
                track.push(Key { time: 0., value });
            }
        } else {
            let aligned_tracks = (clip.tracks.len() + 3) & !3;
            for frame in 0..frames {
                let values = cursor.floats(aligned_tracks)?;
                let time = frame_time(clip, frame);
                for (track, value) in clip.tracks.iter_mut().zip(values) {
                    track.push(Key { time, value });
                }
            }
        }
    }
    Ok(())
}
