//! Static surfaces for environmental, vegetation and display material families.
//!
//! Native texture/tint/coverage inputs are retained. Runtime wind, destruction,
//! screen-space effects and time animation are frozen and reported in notes.

#![expect(
    clippy::cast_precision_loss,
    reason = "texture dimensions are bounded to small u32 values"
)]

use super::*;

pub(super) fn supports(template: &str) -> bool {
    TEMPLATES.contains(&template)
}

const TEMPLATES: &[&str] = &[
    r"base\materials\skin_with_hair.mt",
    r"base\fx\_shaders\snow_pattern.mt",
    r"base\materials\speedtree_3d_v8_seams.mt",
    r"base\materials\speedtree_3d_v8_onesided_gradient_recolor.mt",
    r"base\fx\shaders\metal_base_alpha_animated.mt",
    r"ep1\fx\shaders\animated_emissive.mt",
    r"base\materials\diode_sign.mt",
    r"base\materials\mesh_decal_revealed.mt",
    r"base\materials\decal.remt",
    r"base\materials\mesh_decal_wet_character.mt",
    r"base\materials\ver_skinned_mov_parade.mt",
    r"engine\materials\metal_base_packed.mt",
    r"base\fx\shaders\particles_hologram.mt",
    r"ep1\fx\characters\main_npc\baron\baron_tube_glow.mt",
    r"base\materials\ver_mov_multilayered.mt",
    r"base\fx\shaders\multilayer_alpha_animated.mt",
    r"base\fx\shaders\television_ad.mt",
    r"engine\materials\mirror.remt",
    r"base\materials\ver_mov.mt",
    r"base\materials\glass_window_rain.mt",
    r"base\materials\neon_parallax.mt",
    r"base\materials\cloth_tarps.mt",
    r"base\fx\shaders\big_screen_glitch.mt",
    r"base\materials\glass_flat_twosided.mt",
    r"base\materials\window_interior_uv.mt",
    r"base\fx\_shaders\flat_fog_masked_notxaa.mt",
    r"base\fx\_shaders\flat_fog_masked.mt",
    r"base\materials\decal_forward.mt",
    r"base\materials\mikoshi_parallax.mt",
    r"base\materials\animated_ceiling.mt",
    r"base\fx\_shaders\diode_pavements_ui.mt",
    r"base\fx\materials\holograms\silverhand_props_overlay.mt",
    r"engine\materials\fallback.remt",
    r"engine\materials\transparent_notxaa_2.mt",
    r"base\materials\emissive_control.mt",
    r"engine\materials\metal_base_fence.mt",
    r"base\materials\cable.mt",
    r"base\materials\metal_base_bink.mt",
    r"base\materials\ice_fluid_mov.mt",
    r"base\fx\_shaders\metal_base_scrolling.mt",
    r"base\fx\shaders\water_plane.mt",
    r"base\fx\shaders\diode_pavements.mt",
    r"engine\materials\pbr_simple.mt",
    r"base\fx\_shaders\dirt_animated_masked.mt",
    r"base\fx\shaders\parallaxscreen_transparent_ui_txaa.mt",
    r"base\materials\mirror_glass.mt",
    r"base\materials\earth_globe_atmosphere.mt",
    r"base\materials\earth_globe_lights.mt",
    r"base\materials\starfield.mt",
    r"base\materials\earth_globe.mt",
    r"base\materials\ver_mov_glass.mt",
    r"base\fx\shaders\alphablend_glass.mt",
    r"base\materials\sq021_glass_fluid.mt",
    r"base\materials\glass_blendable.mt",
    r"engine\materials\global_water_patch.mt",
    r"base\materials\q202_lake_surface.mt",
    r"base\materials\speedtree_3d_v8_twosided.mt",
    r"base\materials\speedtree_3d_v8_onesided.mt",
    r"base\materials\ver_skinned_mov.mt",
    r"base\fx\shaders\hud_markers_transparent.mt",
    r"base\fx\shaders\fog_laser.mt",
    r"base\materials\metal_base_trafficlight_proxy.mt",
    r"base\fx\_shaders\shard.mt",
    r"base\fx\shaders\simple_additive_ui.mt",
    r"base\fx\shaders\simple_additive_ui_notxaa.mt",
    r"base\materials\gmpl_pp_screenmask.mt",
    r"base\materials\window_very_long_distance.mt",
    r"base\materials\frosted_glass.mt",
    r"base\materials\window_parallax_interior_proxy.mt",
    r"base\materials\cloth_wind_proxy.mt",
    r"base\materials\proxy_ad.mt",
    r"base\materials\metal_base_det_vc.mt",
    r"base\materials\fur_shell.mt",
    r"ep1\fx\shaders\holographic_waterfall.mt",
    r"base\fx\_shaders\hologram_v2_screenspace_vfx_no_depth.mt",
    r"base\fx\_shaders\hologram_v2_screenspace_vfx.mt",
    r"ep1\fx\shaders\simple_hologram.mt",
    r"ep1\fx\shaders\dirty_emissive.mt",
    r"base\fx\_shaders\hologram_v2.mt",
    r"base\materials\cloth_tarps_pbr_no_vat.mt",
    r"base\materials\cloth_tarps_no_vat.mt",
    r"base\materials\cloth_tarps_pbr.mt",
    r"base\fx\_shaders\q305_core_flares.mt",
    r"base\materials\metal_base_ui_q303.mt",
];

#[expect(clippy::too_many_lines, reason = "explicit native material contracts")]
pub(super) fn bake(
    name: String,
    template: &str,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
    enable_mask: bool,
) -> Result<PbrMaterial, PbrError> {
    let leaf = template.rsplit('\\').next().unwrap_or(template);
    let mut material = match leaf {
        "skin_with_hair.mt" => character::skin(name, record, data, context)?,
        "snow_pattern.mt" => snow(name, record, data, context)?,
        "ver_mov_multilayered.mt"
        | "multilayer_alpha_animated.mt"
        | "silverhand_props_overlay.mt" => bake_multilayered(name, record, data, context)?,
        "cloth_tarps.mt" | "cloth_tarps_no_vat.mt" => {
            let material = bake_multilayered(name, record, data, context)?;
            tarp(material, record, data, context)?
        }
        "cloth_tarps_pbr.mt" | "cloth_tarps_pbr_no_vat.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            material.uv_transform = vector_uv(data, "UVScaleBias", [1., 1., 0., 0.]);
            tarp(material, record, data, context)?
        }
        "mesh_decal_revealed.mt" | "mesh_decal_wet_character.mt" | "decal.remt" => {
            let mut material = bake_decal(name, record, data, context)?;
            let columns = scalar(data, "AnimationFramesWidth", scalar(data, "SubUVx", 1.)).max(1.);
            let rows = scalar(data, "AnimationFramesHeight", scalar(data, "SubUVy", 1.)).max(1.);
            let frame = scalar(data, "TileNumber", scalar(data, "Frame", 0.))
                .max(0.)
                .floor();
            material.uv_transform = [
                scalar(data, "UVScaleX", 1.) / columns,
                scalar(data, "UVScaleY", 1.) / rows,
                scalar(data, "UVOffsetX", 0.) + frame.rem_euclid(columns) / columns,
                scalar(data, "UVOffsetY", 0.) + (frame / columns).floor() / rows,
            ];
            material
        }
        "speedtree_3d_v8_onesided.mt"
        | "speedtree_3d_v8_twosided.mt"
        | "speedtree_3d_v8_seams.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_texture = optional_texture(context.repository, data, "DiffuseMap")?;
            material.normal_texture = optional_texture(context.repository, data, "NormalMap")?;
            material.alpha_mode = "MASK";
            material.alpha_cutoff = Some(0.5);
            let gloss = optional_texture(context.repository, data, "TransGlossMap")?
                .map(|p| load_rgba(&p, context.size))
                .transpose()?;
            if let Some(gloss) = gloss {
                material.metallic_roughness_texture = Some(character::bake_map(
                    record,
                    "foliage-surface.png",
                    context,
                    |size| {
                        rasterize(size, |x, y| {
                            // SpeedTree's gloss is in alpha; RGB describe transmission.
                            Rgba([255, 255 - gloss.get_pixel(x, y)[3], 0, 255])
                        })
                    },
                )?);
                material.roughness_factor = 1.;
            }
            material
        }
        "speedtree_3d_v8_onesided_gradient_recolor.mt" => {
            gradient_surface(name, record, data, context)?
        }
        "metal_base_alpha_animated.mt" | "metal_base_det_vc.mt" => {
            let mut alias = record.clone();
            alias["MaterialTemplate"] = json!(r"base\materials\metal_base_det_dithered.mt");
            bake_material(&alias, context)?
        }
        "animated_emissive.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor = gamma_color(data.get("MainColor"), [1.; 4]);
            material.emissive_factor = gamma_color(data.get("EmissiveColor"), [1.; 4])[..3]
                .try_into()
                .expect("RGB");
            material.emissive_strength = 1.;
            let turns = scalar(data, "NumberOfTurns", 1.);
            let rows = scalar(data, "NumberOfRows", 1.);
            let offset = scalar(data, "Offset", 0.);
            let power = scalar(data, "EmissivePower", 1.).max(0.001);
            material.emissive_texture = Some(character::bake_map(
                record,
                "animated-emissive-static.png",
                context,
                |size| {
                    rasterize(size, |x, y| {
                        let phase = (x as f32 + 0.5) / size as f32 * rows
                            + (y as f32 + 0.5) / size as f32 * turns
                            + offset;
                        let value =
                            unit_to_byte((phase * std::f32::consts::TAU).sin().max(0.).powf(power));
                        Rgba([value, value, value, 255])
                    })
                },
            )?);
            material.preview_notes.push("Legacy animated emission has no installed shader. Static UV bands approximate its authored turn/row density, phase, main color and emission color; exact world-space projection and moving masks are unavailable".into());
            material
        }
        "television_ad.mt" => {
            let mut material = display(name, data, context, "AdTexture", "EmissiveColor")?;
            material.uv_transform = [
                1. / scalar(data, "TilesWidth", 1.).max(1.),
                1. / scalar(data, "TilesHeight", 1.).max(1.),
                0.,
                0.,
            ];
            material
        }
        "diode_sign.mt" => display(name, data, context, "Text", "ColorForeground")?,
        "diode_pavements.mt" | "diode_pavements_ui.mt" => {
            let mut material = display(name, data, context, "SignTexture", "EmissiveColor")?;
            material.alpha_mode = if scalar(data, "UseMaskAsAlphaThreshold", 0.) > 0. {
                "MASK"
            } else {
                "OPAQUE"
            };
            material.alpha_cutoff = (material.alpha_mode == "MASK").then_some(0.5);
            material
        }
        "neon_parallax.mt" => {
            let mut material = bake_metal_base(name, record, data, context, true)?;
            material.emissive_texture = material.base_color_texture.clone();
            material.emissive_factor = material.base_color_factor[..3].try_into().expect("RGB");
            material.emissive_strength = 1.;
            material.uv_transform = uv(data);
            material
        }
        "metal_base_ui_q303.mt" | "simple_additive_ui.mt" | "simple_additive_ui_notxaa.mt" => {
            let mut material = display(name, data, context, "UIRenderTexture", "EmissiveColor")?;
            material.alpha_mode = "BLEND";
            material.uv_transform = vector_uv(data, "UvTilingAndOffset", [1., 1., 0., 0.]);
            material
        }
        "parallaxscreen_transparent_ui_txaa.mt" => bake_screen(name, data, context)?,
        "window_parallax_interior_proxy.mt" | "mikoshi_parallax.mt" => {
            bake_window_parallax(name, data, context)?
        }
        "window_interior_uv.mt" => {
            let mut material = display(name, data, context, "Glass", "GlassColor")?;
            material.emissive_strength = scalar(data, "LightIntensity", 0.).max(0.);
            material.normal_texture = optional_texture(context.repository, data, "Normal")?;
            material
        }
        "glass_window_rain.mt"
        | "ver_mov_glass.mt"
        | "glass_blendable.mt"
        | "frosted_glass.mt"
        | "glass_flat_twosided.mt"
        | "sq021_glass_fluid.mt" => bake_glass(name, record, data, context)?,
        "mirror.remt" | "mirror_glass.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor = [0.92, 0.92, 0.92, 1.];
            material.metallic_factor = 1.;
            material.roughness_factor = 0.025;
            material.normal_texture = optional_texture(context.repository, data, "Normal")?;
            material.preview_notes.push("Mirror reflects the studio environment; live scene/camera reflections are not available".into());
            material
        }
        "metal_base_packed.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            if let Some(path) = optional_texture(context.repository, data, "RoughMetal")? {
                let map = load_rgba(&path, context.size)?;
                material.metallic_roughness_texture = Some(character::bake_map(
                    record,
                    "packed-surface.png",
                    context,
                    |size| {
                        rasterize(size, |x, y| {
                            let pixel = map.get_pixel(x, y);
                            Rgba([
                                255,
                                unit_to_byte((f32::from(pixel[0]) / 255.).mul_add(
                                    scalar(data, "RoughnessScale", 1.),
                                    scalar(data, "RoughnessBias", 0.),
                                )),
                                unit_to_byte((f32::from(pixel[1]) / 255.).mul_add(
                                    scalar(data, "MetalnessScale", 1.),
                                    scalar(data, "MetalnessBias", 0.),
                                )),
                                255,
                            ])
                        })
                    },
                )?);
                material.roughness_factor = 1.;
                material.metallic_factor = 1.;
            }
            material
        }
        "fur_shell.mt" => fur(name, record, data, context)?,
        "pbr_simple.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor = gamma_color(data.get("Color"), [1.; 4]);
            material.roughness_factor = scalar(data, "Roughness", 1.).clamp(0., 1.);
            material.metallic_factor = scalar(data, "Metalness", 0.).clamp(0., 1.);
            material
        }
        "particles_hologram.mt" => {
            let mut parameters = data.clone();
            parameters.insert(
                "Color".into(),
                data.get("ColorParam").cloned().unwrap_or(Value::Null),
            );
            // AlphaSubUV dimensions belong to the alpha mask, not a diffuse map.
            parameters.insert(
                "Diffuse".into(),
                data.get("AlphaMask").cloned().unwrap_or(Value::Null),
            );
            parameters.insert(
                "SubUVWidth".into(),
                json!(scalar(data, "AlphaSubUVWidth", 1.)),
            );
            parameters.insert(
                "SubUVHeight".into(),
                json!(scalar(data, "AlphaSubUVHeight", 1.)),
            );
            bake_particle_fx(name, record, &parameters, context)?
        }
        "hud_markers_transparent.mt" | "transparent_notxaa_2.mt" | "alphablend_glass.mt" => {
            let mut material = bake_particle_fx(name, record, data, context)?;
            material.uv_transform = [
                scalar(data, "TillingX", 1.),
                scalar(data, "TillingY", 1.),
                scalar(data, "OffsetX", 0.),
                scalar(data, "OffsetY", 0.),
            ];
            material
        }
        "hologram_v2.mt"
        | "hologram_v2_screenspace_vfx.mt"
        | "hologram_v2_screenspace_vfx_no_depth.mt" => {
            let mut material = display(name, data, context, "Diffuse", "Color_Light")?;
            material.alpha_mode = "BLEND";
            material.base_color_factor[3] = 0.65;
            material.normal_texture = optional_texture(context.repository, data, "Normal")?;
            material
        }
        "simple_hologram.mt" => display(name, data, context, "Scanline", "TopColor")?,
        "holographic_waterfall.mt" | "baron_tube_glow.mt" => {
            let mut material = PbrMaterial::new(name);
            let color = gamma_color(data.get("ColorLight"), [1.; 4]);
            material.base_color_factor = [0., 0., 0., 0.65];
            material.emissive_factor = color[..3].try_into().expect("RGB");
            material.emissive_strength = 1.;
            material.alpha_mode = "BLEND";
            material
        }
        "earth_globe_atmosphere.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor =
                gamma_color(data.get("AtmosphereColor"), [0.2, 0.4, 0.8, 0.15]);
            material.base_color_factor[3] = 0.15;
            material.alpha_mode = "BLEND";
            material
        }
        "earth_globe_lights.mt" => display(name, data, context, "DiffuseTexture", "DiffuseColor")?,
        "starfield.mt" => display(name, data, context, "Galaxy", "Color")?,
        "water_plane.mt"
        | "global_water_patch.mt"
        | "q202_lake_surface.mt"
        | "ice_fluid_mov.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor = gamma_color(
                data.get("TintColor")
                    .or_else(|| data.get("ScatteringColor"))
                    .or_else(|| data.get("WaterColorDeep")),
                [0.03, 0.12, 0.15, 1.],
            );
            material.base_color_factor[3] = scalar(data, "WaterOpacity", 0.65).clamp(0.05, 1.);
            material.normal_texture = optional_texture(context.repository, data, "Normal")?;
            material.roughness_factor = scalar(data, "WaterRoughness", 0.05).clamp(0., 1.);
            material.transmission = 0.65;
            material.alpha_mode = "BLEND";
            material
        }
        "dirt_animated_masked.mt" => {
            masked_emission(name, record, data, context, "Dirt", "Mask", "DirtColor")?
        }
        "dirty_emissive.mt" => {
            masked_emission(name, record, data, context, "Dirt", "Dirt", "BaseColor")?
        }
        "metal_base_scrolling.mt" => {
            masked_emission(name, record, data, context, "Mask", "Mask", "Bright")?
        }
        "fog_laser.mt" => masked_emission(
            name,
            record,
            data,
            context,
            "NoiseTexture",
            "NoiseTexture",
            "LaserColor",
        )?,
        "flat_fog_masked.mt" | "flat_fog_masked_notxaa.mt" => masked_emission(
            name,
            record,
            data,
            context,
            "RefractionMask",
            "RefractionMask",
            "CrackColor",
        )?,
        "q305_core_flares.mt" => display(name, data, context, "GlareTexture", "Color")?,
        "animated_ceiling.mt" => display(name, data, context, "ColorGradientTex", "Color")?,
        "ver_mov.mt"
        | "ver_skinned_mov.mt"
        | "ver_skinned_mov_parade.mt"
        | "metal_base_fence.mt"
        | "cable.mt"
        | "metal_base_bink.mt"
        | "metal_base_trafficlight_proxy.mt"
        | "cloth_wind_proxy.mt"
        | "proxy_ad.mt"
        | "emissive_control.mt"
        | "window_very_long_distance.mt"
        | "earth_globe.mt"
        | "big_screen_glitch.mt"
        | "decal_forward.mt"
        | "shard.mt"
        | "gmpl_pp_screenmask.mt"
        | "fallback.remt" => bake_metal_base(
            name,
            record,
            data,
            context,
            enable_mask || leaf == "metal_base_fence.mt",
        )?,
        _ => return Err(PbrError::UnsupportedTemplate(template.to_owned())),
    };
    material.preview_notes.push(format!("Static {leaf} preview retains authored surface inputs; runtime animation, wind/destruction deformation, screen-space projection and simulation are not evaluated"));
    Ok(material)
}

fn snow(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let rows = scalar(data, "AmountOfRows", 350.).max(1.);
    let columns = scalar(data, "AmountOfColumns", 400.).max(1.);
    let intensity = scalar(data, "Multiplier", 0.25).clamp(0., 1.);
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(character::bake_map(
        record,
        "snow-time-zero.png",
        context,
        |size| {
            rasterize(size, |x, y| {
                let u = (x as f32 + 0.5) / size as f32 * columns;
                let v = (y as f32 + 0.5) / size as f32 * rows;
                let seed = u.floor().to_bits().wrapping_mul(0x9e37_79b9)
                    ^ v.floor().to_bits().wrapping_mul(0x85eb_ca6b);
                let noise = (seed ^ (seed >> 16)).wrapping_mul(0x7feb_352d);
                let star = if noise % 97 == 0 { intensity } else { 0. };
                Rgba([255, 255, 255, unit_to_byte(star)])
            })
        },
    )?);
    material.emissive_texture = material.base_color_texture.clone();
    material.emissive_factor = [1.; 3];
    material.emissive_strength = 1.;
    material.alpha_mode = "BLEND";
    material.preview_notes.push("Legacy snow_pattern has no installed shader template or texture. Preview uses a deterministic static star field at the authored row/column density and intensity; exact runtime noise and motion are unavailable".into());
    Ok(material)
}

fn gradient_surface(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = bake_metal_base(name, record, data, context, true)?;
    let base = material
        .base_color_texture
        .as_deref()
        .map(|p| load_rgba(p, context.size))
        .transpose()?;
    let gradient = optional_texture(context.repository, data, "GradientMap")?
        .map(image::open)
        .transpose()?
        .map(image::DynamicImage::into_rgba8);
    let mask = optional_texture(context.repository, data, "Mask")?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    if let Some(gradient) = gradient {
        material.base_color_texture = Some(character::bake_map(
            record,
            "vegetation-gradient.png",
            context,
            |size| {
                rasterize(size, |x, y| {
                    let input = base
                        .as_ref()
                        .map_or(Rgba([255; 4]), |image| *image.get_pixel(x, y));
                    let coordinate = f32::from(input[0]) / 255. * (gradient.width() - 1) as f32;
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "coordinate is bounded by image width"
                    )]
                    let left = coordinate.floor() as u32;
                    let right = (left + 1).min(gradient.width() - 1);
                    let blend = coordinate - left as f32;
                    let a = gradient.get_pixel(left, gradient.height() / 2);
                    let b = gradient.get_pixel(right, gradient.height() / 2);
                    let coverage = mask
                        .as_ref()
                        .map_or(1., |image| f32::from(image.get_pixel(x, y)[0]) / 255.);
                    let mut output = input;
                    for channel in 0..3 {
                        let mapped = lerp(
                            srgb_to_linear(f32::from(a[channel]) / 255.),
                            srgb_to_linear(f32::from(b[channel]) / 255.),
                            blend,
                        );
                        output[channel] = unit_to_byte(linear_to_srgb(lerp(
                            srgb_to_linear(f32::from(input[channel]) / 255.),
                            mapped,
                            coverage,
                        )));
                    }
                    output
                })
            },
        )?);
    }
    let tile = scalar(data, "LayerTile", 1.);
    material.uv_transform = [tile, tile, 0., 0.];
    Ok(material)
}

fn uv(data: &Map<String, Value>) -> [f32; 4] {
    [
        scalar(data, "UvTilingX", 1.),
        scalar(data, "UvTilingY", 1.),
        scalar(data, "UvOffsetX", 0.),
        scalar(data, "UvOffsetY", 0.),
    ]
}

fn vector_uv(data: &Map<String, Value>, key: &str, default: [f32; 4]) -> [f32; 4] {
    data.get(key).map_or(default, |v| {
        [
            scalar_object(v, "X", default[0]),
            scalar_object(v, "Y", default[1]),
            scalar_object(v, "Z", default[2]),
            scalar_object(v, "W", default[3]),
        ]
    })
}

fn display(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
    texture: &str,
    tint: &str,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = optional_texture(context.repository, data, texture)?;
    material.emissive_texture = material.base_color_texture.clone();
    let color = gamma_color(data.get(tint), [1.; 4]);
    material.base_color_factor = [0., 0., 0., color[3]];
    material.emissive_factor = color[..3].try_into().expect("RGB");
    // Studio exposure normalization keeps the authored hue legible.
    material.emissive_strength = 1.;
    material.roughness_factor = scalar(data, "Roughness", 0.5).clamp(0., 1.);
    material.metallic_factor = scalar(data, "Metalness", 0.).clamp(0., 1.);
    material
        .preview_notes
        .push("Emissive exposure is normalized for the studio camera".into());
    Ok(material)
}

fn tarp(
    mut material: PbrMaterial,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let tear = optional_texture(context.repository, data, "TearMap")?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    if let Some(tear) = tear {
        let base = material
            .base_color_texture
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        let tile = vector_uv(data, "TearMapTiling", [1., 1., 0., 0.]);
        let contrast = scalar(data, "TearContrast", 1.);
        let offset = scalar(data, "TearOffset", 0.);
        material.base_color_texture = Some(character::bake_map(
            record,
            "tarp-coverage.png",
            context,
            |size| {
                rasterize(size, |x, y| {
                    let mut pixel = base
                        .as_ref()
                        .map_or(Rgba([255; 4]), |image| *image.get_pixel(x, y));
                    let mask = sample_repeat(
                        &tear,
                        (x as f32 + 0.5) / size as f32 * tile[0],
                        (y as f32 + 0.5) / size as f32 * tile[1],
                    )[3];
                    pixel[3] = unit_to_byte(
                        f32::from(pixel[3]) / 255. * mask.mul_add(contrast, offset).clamp(0., 1.),
                    );
                    pixel
                })
            },
        )?);
        material.alpha_mode = "MASK";
        material.alpha_cutoff = Some(scalar(data, "AlphaThreshold", 0.5).max(0.01));
    }
    Ok(material)
}

fn fur(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut parameters = data.clone();
    if let Some(base) = data.get("BaseMap") {
        parameters.insert("BaseColor".into(), base.clone());
    }
    let mut material = gradient_surface(name, record, &parameters, context)?;
    material.alpha_mode = "MASK";
    material.alpha_cutoff = Some(scalar(data, "AlphaThreshold", 0.5).max(0.01));
    let alpha = optional_texture(context.repository, data, "AlphaMap")?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    if let Some(alpha) = alpha {
        let base = material
            .base_color_texture
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        material.base_color_texture = Some(character::bake_map(
            record,
            "fur-coverage.png",
            context,
            |size| {
                rasterize(size, |x, y| {
                    let mut pixel = base
                        .as_ref()
                        .map_or(Rgba([255; 4]), |image| *image.get_pixel(x, y));
                    pixel[3] = alpha.get_pixel(x, y)[0];
                    pixel
                })
            },
        )?);
    }
    material.preview_notes.push("Fur retains the authored mesh, gradient recoloring, surface and alpha maps; procedural shell extrusion, wind and strand self-shadowing are approximated by the static surface".into());
    Ok(material)
}

fn masked_emission(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
    texture: &str,
    mask: &str,
    tint: &str,
) -> Result<PbrMaterial, PbrError> {
    let mut material = display(name, data, context, texture, tint)?;
    let color = material
        .base_color_texture
        .as_deref()
        .map(|p| load_rgba(p, context.size))
        .transpose()?;
    let mask = optional_texture(context.repository, data, mask)?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    material.base_color_texture = Some(character::bake_map(
        record,
        "effect-coverage.png",
        context,
        |size| {
            rasterize(size, |x, y| {
                let mut pixel = color
                    .as_ref()
                    .map_or(Rgba([255; 4]), |image| *image.get_pixel(x, y));
                pixel[3] = unit_to_byte(
                    f32::from(pixel[3]) / 255.
                        * mask
                            .as_ref()
                            .map_or(1., |image| f32::from(image.get_pixel(x, y)[0]) / 255.),
                );
                pixel
            })
        },
    )?);
    material.emissive_texture = material.base_color_texture.clone();
    material.alpha_mode = "BLEND";
    Ok(material)
}
