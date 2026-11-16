use super::{
    BakeContext, PbrError, PbrMaterial, bake_lock, cache_directory, depot_json_path, gamma_color,
    load_rgba, malformed, optional_texture, pack_metallic_roughness, required_resource, root_chunk,
    save_png, scalar, unit_to_byte,
};
use image::{Rgba, RgbaImage};
use serde_json::{Map, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(super) fn skin(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let albedo = Some(required_map(context, data, "Albedo")?);
    let secondary = if scalar(data, "SecondaryAlbedoInfluence", 0.) == 0. {
        None
    } else {
        Some(required_map(context, data, "SecondaryAlbedo")?)
    };
    let mask = map(context, data, "TintColorMask")?;
    let tint = gamma_color(data.get("TintColor"), [1.; 4]);
    let strength = scalar(data, "TintScale", 0.).clamp(0., 1.);
    let secondary_strength = scalar(data, "SecondaryAlbedoInfluence", 0.).clamp(0., 1.);
    let secondary_tint = scalar(data, "SecondaryAlbedoTintColorInfluence", 1.).clamp(0., 1.);
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(bake_map(record, "skin-color.png", context, |size| {
        image(size, |x, y| {
            let base = linear_pixel(albedo.as_ref(), x, y, [1.; 4]);
            let secondary = linear_pixel(secondary.as_ref(), x, y, base);
            let amount = sample(mask.as_ref(), x, y, 0, 1.) * strength;
            let mut result = [0.; 4];
            for channel in 0..3 {
                let base = base[channel] * (1. + (tint[channel] - 1.) * amount);
                let overlay =
                    secondary[channel] * (1. + (tint[channel] - 1.) * amount * secondary_tint);
                result[channel] = base + (overlay - base) * secondary[3] * secondary_strength;
            }
            result[3] = 1.;
            encode(result)
        })
    })?);
    material.normal_texture = optional_texture(context.repository, data, "Normal")?;
    surface(&mut material, record, data, context)?;
    material.preview_notes.push("Skin uses authored albedo, tint mask, primary normal and roughness; subsurface scattering, micro/detail normals, wrinkle and blood-flow layers are omitted".into());
    if let Some(mask) = optional_texture(context.repository, data, "EmissiveMask")? {
        material.emissive_texture = Some(mask);
        material.emissive_factor = [1.; 3];
        material.emissive_strength = 2_f32.powf(scalar(data, "EmissiveEV", 0.).clamp(-20., 20.));
    }
    Ok(material)
}

pub(super) fn skin_morph(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let amount = scalar(data, "Morph", 0.).clamp(0., 1.);
    let mut source = data.clone();
    // Catalogue views sample a stable endpoint. Runtime transition masks and
    // animation are meaningful only with the game's changing morph parameter.
    if amount >= 0.5 {
        for (key, target) in [
            ("Albedo", "AlbedoTarget"),
            ("Normal", "NormalTarget"),
            ("TintColor", "TintColorTarget"),
            ("TintScale", "TintScaleTarget"),
        ] {
            if let Some(value) = data.get(target) {
                source.insert(key.into(), value.clone());
            }
        }
    }
    let mut material = skin(name, record, &source, context)?;
    material.preview_notes.push(format!("Morph skin displays the authored {} endpoint at Morph={amount}; animated masked transition is frozen", if amount >= 0.5 { "target" } else { "source" }));
    Ok(material)
}

pub(super) fn hair_proxy(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = optional_texture(context.repository, data, "Albedo")?;
    material.normal_texture = optional_texture(context.repository, data, "Normal")?;
    surface(&mut material, record, data, context)?;
    material.metallic_factor = scalar(data, "MetalnessBias", 0.).clamp(0., 1.);
    material.alpha_mode = "BLEND";
    material.stochastic_coverage = true;
    material.preview_notes.push("Hair proxy uses its authored albedo coverage, normal and roughness; native scattering is approximated by studio surface lighting".into());
    Ok(material)
}

pub(super) fn transparent_emission(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let diffuse = map(context, data, "Diffuse")?;
    let mask = map(context, data, "Mask")?;
    let animation = map(context, data, "AnimMask")?;
    let mut material = PbrMaterial::new(name);
    let texture = bake_map(record, "transparent-emission.png", context, |size| {
        image(size, |x, y| {
            let mut pixel = linear_pixel(diffuse.as_ref(), x, y, [1.; 4]);
            pixel[3] *=
                sample(mask.as_ref(), x, y, 0, 1.) * sample(animation.as_ref(), x, y, 0, 1.);
            encode(pixel)
        })
    })?;
    material.base_color_texture = Some(texture.clone());
    material.base_color_factor = [0., 0., 0., 1.];
    material.emissive_texture = Some(texture);
    material.emissive_factor = gamma_color(data.get("EmissiveColor"), [1.; 4])[..3]
        .try_into()
        .expect("RGB");
    material.emissive_strength = 1.;
    material.alpha_mode = "BLEND";
    material.preview_notes.push("Transparent emission retains diffuse color and both authored coverage masks; animated mask is frozen and additive compositing uses alpha blending".into());
    Ok(material)
}

pub(super) fn holo_mask(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let diffuse = required_map(context, data, "Diffuse")?;
    let low = gamma_color(data.get("MinColor"), [1.; 4]);
    let high = gamma_color(data.get("MaxColor"), [0., 1., 0.5, 1.]);
    let eyes = gamma_color(data.get("EyesColor"), [1., 0., 0., 1.]);
    let mut material = PbrMaterial::new(name);
    let texture = bake_map(record, "holo-mask.png", context, |size| {
        image(size, |x, y| {
            let pixel = super::sample_repeat(
                &diffuse,
                (x as f32 + 0.5) / size as f32,
                (y as f32 + 0.5) / size as f32,
            );
            let mut result = [0.; 4];
            for channel in 0..3 {
                result[channel] =
                    eyes[channel] * pixel[0] + low[channel] * pixel[1] + high[channel] * pixel[2];
            }
            result[3] = pixel[0].max(pixel[1]).max(pixel[2]) * pixel[3];
            encode(result)
        })
    })?;
    material.base_color_texture = Some(texture.clone());
    material.base_color_factor = [0., 0., 0., 1.];
    material.emissive_texture = Some(texture);
    material.emissive_factor = [1.; 3];
    material.emissive_strength = 1.;
    material.alpha_mode = "BLEND";
    material.double_sided = true;
    material.preview_notes.push("Holographic mask preserves the complete authored diffuse UVs, RGB color masks and alpha silhouette; scanlines, tearing and animation are frozen, emission exposure is normalized".into());
    Ok(material)
}

pub(super) fn hair(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let id = Some(required_map(context, data, "Strand_ID")?);
    let root_to_tip = Some(required_map(context, data, "Strand_Gradient")?);
    let alpha = Some(required_map(context, data, "Strand_Alpha")?);
    let (id_colors, tip_colors) = if data
        .get("HairProfile")
        .and_then(super::resource_string)
        .is_some()
    {
        let profile = resource_json(context.repository, data, "HairProfile")?;
        let profile = root_chunk(&profile)?;
        (
            gradient(&profile["gradientEntriesID"])?,
            gradient(&profile["gradientEntriesRootToTip"])?,
        )
    } else {
        // Native eyelash instances leave the profile unbound. Their shader's
        // black colour still uses the authored strand alpha and flow textures.
        (vec![(0., [0., 0., 0., 1.])], vec![(0., [1.; 4])])
    };
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(bake_map(record, "hair-color.png", context, |size| {
        image(size, |x, y| {
            let first = gradient_sample(&id_colors, sample(id.as_ref(), x, y, 0, 0.5));
            let second = gradient_sample(&tip_colors, sample(root_to_tip.as_ref(), x, y, 0, 0.5));
            let color = [
                first[0] * second[0],
                first[1] * second[1],
                first[2] * second[2],
                sample(alpha.as_ref(), x, y, 0, 1.),
            ];
            encode(color)
        })
    })?);
    material.roughness_factor = (0.35 * scalar(data, "RoughnessScale", 1.)
        + scalar(data, "RoughnessBias", 0.))
    .clamp(0.04, 1.);
    let cutoff = scalar(data, "AlphaCutoff", 0.).clamp(0., 1.);
    material.alpha_mode = if cutoff > 0. { "MASK" } else { "BLEND" };
    material.alpha_cutoff = (cutoff > 0.).then_some(cutoff);
    material.stochastic_coverage = cutoff == 0.;
    material.hair = Some(super::HairShading {
        flow_texture: optional_texture(context.repository, data, "Flow")?,
        flow_strength: scalar(data, "FlowStrength", 1.).clamp(0., 1.),
        vertex_color_strength: scalar(data, "VertexColorStrength", 0.).clamp(0., 1.),
    });
    material.preview_notes.push("Hair uses profile gradients, strand coverage, authored flow and vertex-color strength. Directional strand lighting approximates native scattering; multiple scattering and cast shadows are omitted".into());
    Ok(material)
}

/// Bakes the authored grayscale-to-color lookup and decal coverage independently.
pub(super) fn gradient_decal(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = super::bake_decal(name, record, data, context)?;
    let diffuse = required_map(context, data, "DiffuseTexture")?;
    let gradient_path = optional_texture(context.repository, data, "GradientMap")?
        .ok_or_else(|| malformed("missing GradientMap"))?;
    let gradient = image::open(gradient_path)?.into_rgba8();
    let mask = map(context, data, "MaskTexture")?;
    let secondary = map(context, data, "SecondaryMask")?;
    let enable_mask = record
        .get("EnableMask")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let influence = scalar(data, "SecondaryMaskInfluence", 0.).clamp(0., 1.);
    let uv_scale = scalar(data, "SecondaryMaskUVScale", 1.);
    if !uv_scale.is_finite() {
        return Err(malformed("nonfinite decal mask UV scale"));
    }
    material.base_color_texture = Some(bake_map(record, "gradient-decal.png", context, |size| {
        image(size, |x, y| {
            let input = diffuse.get_pixel(x, y);
            let position = u32::from(input[0]) * (gradient.width() - 1);
            let left = position / 255;
            let right = (left + 1).min(gradient.width() - 1);
            let blend =
                f32::from(u8::try_from(position % 255).expect("modulo 255 fits a byte")) / 255.;
            let first = gradient.get_pixel(left, gradient.height() / 2);
            let second = gradient.get_pixel(right, gradient.height() / 2);
            let coverage = if enable_mask {
                sample(mask.as_ref(), x, y, 0, f32::from(input[3]) / 255.)
            } else {
                ((f32::from(input[0]) / 255. - 0.004) / 0.036).clamp(0., 1.)
            };
            let secondary = secondary.as_ref().map_or(1., |map| {
                let sx = scaled_coordinate(x, size, uv_scale);
                let sy = scaled_coordinate(y, size, uv_scale);
                f32::from(map.get_pixel(sx, sy)[0]) / 255.
            });
            let mut color = [0.; 4];
            for channel in 0..3 {
                let a = (f32::from(first[channel]) / 255.).powf(2.2);
                let b = (f32::from(second[channel]) / 255.).powf(2.2);
                color[channel] = a + (b - a) * blend;
            }
            color[3] = coverage * (1. - influence + influence * secondary);
            encode(color)
        })
    })?);
    material.preview_notes.push("Gradient-map decal bakes grayscale color lookup and primary/secondary coverage. Native surface-blending controls are retained for the renderer; standard glTF cannot express receiving-surface composition".into());
    Ok(material)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "wrapped finite texture coordinates are bounded by the image size"
)]
fn scaled_coordinate(pixel: u32, size: u32, scale: f32) -> u32 {
    ((pixel as f32 * scale).rem_euclid(size as f32) as u32).min(size - 1)
}

pub(super) fn eye(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let albedo = Some(required_map(context, data, "Albedo")?);
    let mask = Some(required_map(context, data, "IrisMask")?);
    let profile = resource_json(context.repository, data, "IrisColorGradient")?;
    let colors = gradient(&root_chunk(&profile)?["gradientEntries"])?;
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(bake_map(record, "eye-color.png", context, |size| {
        image(size, |x, y| {
            let base = linear_pixel(albedo.as_ref(), x, y, [1.; 4]);
            let gray = sample(albedo.as_ref(), x, y, 0, 1.);
            let iris = gradient_sample(&colors, gray);
            let amount = sample(mask.as_ref(), x, y, 0, 1.);
            let mut color = base;
            for channel in 0..3 {
                color[channel] += (iris[channel] - base[channel]) * amount;
            }
            color[3] = 1.;
            encode(color)
        })
    })?);
    material.normal_texture = optional_texture(context.repository, data, "Normal")?;
    surface(&mut material, record, data, context)?;
    material.preview_notes.push("Eye iris gradient, mask, albedo, primary normal and roughness are retained; refraction, iris parallax, bubble normals and environment blick are omitted".into());
    Ok(material)
}

pub(super) fn eye_plain(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(
        optional_texture(context.repository, data, "Albedo")?
            .ok_or_else(|| malformed("missing eye Albedo"))?,
    );
    material.normal_texture = optional_texture(context.repository, data, "Normal")?;
    surface(&mut material, record, data, context)?;
    material.preview_notes.push("Native eye albedo, normal and roughness are retained; refraction and iris parallax are approximated".into());
    Ok(material)
}

pub(super) fn eye_shadow(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mask = Some(required_map(context, data, "Mask")?);
    let color = gamma_color(data.get("ShadowColor"), [0., 0., 0., 1.]);
    let exponent = scalar(data, "Exponent", 1.).max(0.001);
    let strength = scalar(data, "Intensity", 1.).max(0.);
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(bake_map(record, "eye-shadow.png", context, |size| {
        image(size, |x, y| {
            encode([
                color[0],
                color[1],
                color[2],
                sample(mask.as_ref(), x, y, 0, 1.).powf(exponent) * strength,
            ])
        })
    })?);
    material.alpha_mode = "BLEND";
    material.roughness_factor = scalar(data, "WetnessRoughness", 0.5).clamp(0., 1.);
    material.preview_notes.push(
        "Eye-shadow additive/subsurface wetness is approximated with masked alpha coverage".into(),
    );
    Ok(material)
}

fn surface(
    material: &mut PbrMaterial,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<(), PbrError> {
    if let Some(roughness) = optional_texture(context.repository, data, "Roughness")? {
        material.metallic_roughness_texture = Some(pack_metallic_roughness(
            record,
            Some(&roughness),
            None,
            scalar(data, "RoughnessScale", 1.),
            scalar(data, "RoughnessBias", 0.),
            0.,
            0.,
            context,
        )?);
        material.metallic_factor = 1.;
        material.roughness_factor = 1.;
    } else {
        material.roughness_factor = 0.45;
    }
    Ok(())
}

fn map(
    context: &BakeContext<'_>,
    data: &Map<String, Value>,
    field: &str,
) -> Result<Option<Arc<RgbaImage>>, PbrError> {
    optional_texture(context.repository, data, field)?
        .map(|p| load_rgba(&p, context.size))
        .transpose()
}

fn required_map(
    context: &BakeContext<'_>,
    data: &Map<String, Value>,
    field: &str,
) -> Result<Arc<RgbaImage>, PbrError> {
    map(context, data, field)?.ok_or_else(|| {
        malformed(format!(
            "character material is missing required {field} texture"
        ))
    })
}

fn resource_json(
    repository: &Path,
    data: &Map<String, Value>,
    field: &str,
) -> Result<Value, PbrError> {
    let path = depot_json_path(repository, required_resource(data, field)?);
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn gradient(value: &Value) -> Result<Vec<(f32, [f32; 4])>, PbrError> {
    let values = value
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| malformed("empty character color gradient"))?;
    let mut stops = values
        .iter()
        .map(|v| {
            let point = scalar(
                v.as_object()
                    .ok_or_else(|| malformed("invalid gradient entry"))?,
                "value",
                0.,
            );
            if !point.is_finite() {
                return Err(malformed("nonfinite gradient position"));
            }
            Ok((point, gamma_color(v.get("color"), [1.; 4])))
        })
        .collect::<Result<Vec<_>, PbrError>>()?;
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(stops)
}

fn gradient_sample(stops: &[(f32, [f32; 4])], position: f32) -> [f32; 4] {
    let index = stops.partition_point(|(point, _)| *point <= position);
    if index == 0 {
        return stops[0].1;
    }
    if index == stops.len() {
        return stops[index - 1].1;
    }
    let (left, right) = (stops[index - 1], stops[index]);
    let amount = if right.0 > left.0 {
        (position - left.0) / (right.0 - left.0)
    } else {
        1.
    };
    std::array::from_fn(|i| left.1[i] + (right.1[i] - left.1[i]) * amount)
}

fn sample(map: Option<&Arc<RgbaImage>>, x: u32, y: u32, channel: usize, default: f32) -> f32 {
    map.map_or(default, |map| {
        f32::from(map.get_pixel(x, y)[channel]) / 255.
    })
}

fn linear_pixel(map: Option<&Arc<RgbaImage>>, x: u32, y: u32, default: [f32; 4]) -> [f32; 4] {
    std::array::from_fn(|i| {
        let value = sample(map, x, y, i, default[i]);
        if i < 3 { value.powf(2.2) } else { value }
    })
}

fn encode(color: [f32; 4]) -> Rgba<u8> {
    Rgba(std::array::from_fn(|i| {
        unit_to_byte(if i < 3 {
            color[i].max(0.).powf(1. / 2.2)
        } else {
            color[i]
        })
    }))
}

fn image(size: u32, pixel: impl Fn(u32, u32) -> Rgba<u8> + Sync) -> RgbaImage {
    super::rasterize(size, pixel)
}

pub(super) fn bake_map(
    record: &Value,
    filename: &str,
    context: &mut BakeContext<'_>,
    build: impl FnOnce(u32) -> RgbaImage,
) -> Result<PathBuf, PbrError> {
    let output = cache_directory(context.repository, record, context.size)?.join(filename);
    let lock = bake_lock(&output);
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if output.is_file() {
        context.reused += 1;
        return Ok(output);
    }
    let result = build(context.size);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    save_png(&result, &output)?;
    context.generated += 1;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_decal_uses_color_lookup_and_both_coverage_masks() {
        let directory = tempfile::tempdir().unwrap();
        RgbaImage::from_fn(2, 2, |x, _| Rgba([if x == 0 { 0 } else { 255 }, 0, 0, 255]))
            .save(directory.path().join("diffuse.png"))
            .unwrap();
        RgbaImage::from_fn(2, 2, |x, _| {
            if x == 0 {
                Rgba([90, 40, 10, 255])
            } else {
                Rgba([20, 10, 5, 255])
            }
        })
        .save(directory.path().join("gradient.png"))
        .unwrap();
        RgbaImage::from_fn(2, 2, |x, y| {
            Rgba([if x == 1 && y == 0 { 0 } else { 255 }; 4])
        })
        .save(directory.path().join("mask.png"))
        .unwrap();
        RgbaImage::from_fn(2, 2, |x, y| {
            Rgba([if x == 0 && y == 1 { 0 } else { 255 }; 4])
        })
        .save(directory.path().join("secondary.png"))
        .unwrap();
        let mut context = BakeContext {
            repository: directory.path(),
            size: 2,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"cap","MaterialTemplate":r"base\materials\mesh_decal_gradientmap_recolor.mt","EnableMask":true,"Data":{
            "DiffuseTexture":"diffuse.xbm","GradientMap":"gradient.xbm","MaskTexture":"mask.xbm","SecondaryMask":"secondary.xbm","SecondaryMaskInfluence":1.
        }});
        let material = crate::pbr::bake_material(&record, &mut context).unwrap();
        let pixels = image::open(material.base_color_texture.unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(pixels.get_pixel(0, 0).0, [90, 40, 10, 255]);
        assert_eq!(pixels.get_pixel(1, 1).0, [20, 10, 5, 255]);
        assert_eq!(pixels.get_pixel(1, 0).0[3], 0);
        assert_eq!(pixels.get_pixel(0, 1).0[3], 0);
    }
    use serde_json::json;

    #[test]
    fn character_materials_preserve_tints_profiles_and_strand_coverage() {
        let directory = tempfile::tempdir().unwrap();
        RgbaImage::from_pixel(2, 2, Rgba([180, 100, 60, 255]))
            .save(directory.path().join("albedo.png"))
            .unwrap();
        RgbaImage::from_fn(2, 2, |x, _| {
            if x == 0 {
                Rgba([0, 0, 0, 255])
            } else {
                Rgba([255; 4])
            }
        })
        .save(directory.path().join("mask.png"))
        .unwrap();
        let mut context = BakeContext {
            repository: directory.path(),
            size: 2,
            generated: 0,
            reused: 0,
        };
        let record = json!({"Name":"skin","MaterialTemplate":r"base\materials\skin.mt","Data":{"Albedo":"albedo.xbm","TintColorMask":"mask.xbm","TintScale":1.,"TintColor":{"Red":255,"Green":0,"Blue":0,"Alpha":255}}});
        let skin = crate::pbr::bake_material(&record, &mut context).unwrap();
        let pixels = image::open(skin.base_color_texture.unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(pixels.get_pixel(0, 0).0, [180, 100, 60, 255]);
        assert_eq!(pixels.get_pixel(1, 0).0, [180, 0, 0, 255]);
        let red = json!([{"value":0.,"color":{"Red":255,"Green":0,"Blue":0,"Alpha":255}}]);
        let white = json!([{"value":0.,"color":{"Red":255,"Green":255,"Blue":255,"Alpha":255}}]);
        fs::write(directory.path().join("red.hp.json"),serde_json::to_vec(&json!({"Data":{"RootChunk":{"gradientEntriesID":red,"gradientEntriesRootToTip":white}}})).unwrap()).unwrap();
        let record = json!({"Name":"hair","MaterialTemplate":r"base\materials\hair.mt","Data":{"Strand_ID":"mask.xbm","Strand_Gradient":"mask.xbm","Strand_Alpha":"mask.xbm","HairProfile":"red.hp","AlphaCutoff":0.}});
        let hair = crate::pbr::bake_material(&record, &mut context).unwrap();
        assert_eq!(hair.alpha_mode, "BLEND");
        let pixels = image::open(hair.base_color_texture.unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(pixels.get_pixel(0, 0).0[3], 0);
        assert_eq!(pixels.get_pixel(1, 0).0, [255, 0, 0, 255]);
        fs::write(
            directory.path().join("red.gradient.json"),
            serde_json::to_vec(&json!({"Data":{"RootChunk":{"gradientEntries":red}}})).unwrap(),
        )
        .unwrap();
        let record = json!({"Name":"eye","MaterialTemplate":r"base\materials\eye_gradient.mt","Data":{"Albedo":"albedo.xbm","IrisMask":"mask.xbm","IrisColorGradient":"red.gradient"}});
        let eye = crate::pbr::bake_material(&record, &mut context).unwrap();
        let pixels = image::open(eye.base_color_texture.unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(pixels.get_pixel(0, 0).0, [180, 100, 60, 255]);
        assert_eq!(pixels.get_pixel(1, 0).0, [255, 0, 0, 255]);
        let missing = json!({"Name":"bad","MaterialTemplate":r"base\materials\hair.mt","Data":{}});
        assert!(crate::pbr::bake_material(&missing, &mut context).is_err());
    }
}
