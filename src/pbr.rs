//! Standard PBR material baking for selected Cyberpunk mesh appearances.

use image::{ImageBuffer, ImageEncoder, ImageError, Rgba, RgbaImage};
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use sha1::{Digest, Sha1};
use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::{Component, Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};
use thiserror::Error;

mod character;
mod environment;

const GLB_MAGIC: u32 = 0x4654_6c67;
const GLB_JSON_CHUNK: u32 = 0x4e4f_534a;
const GLB_BIN_CHUNK: u32 = 0x004e_4942;
const DEFAULT_BAKE_SIZE: u32 = 512;

static BAKE_LOCKS: LazyLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

type ScaledTextureKey = (PathBuf, u32, u64, std::time::SystemTime);

#[derive(Default)]
struct ScaledTextures {
    entries: HashMap<ScaledTextureKey, Arc<RgbaImage>>,
    bytes: usize,
}

static SCALED_TEXTURES: LazyLock<Mutex<ScaledTextures>> =
    LazyLock::new(|| Mutex::new(ScaledTextures::default()));

// This pool only executes pure pixel work. Mesh jobs must never be stolen while
// a bake owner holds its cache lock and waits for the pixel work to finish.
// Rayon callers stay serial: a cross-pool install can execute their outer jobs.
static PIXEL_POOL: LazyLock<Option<rayon::ThreadPool>> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |count| count.get().min(32)))
        .thread_name(|index| format!("ghostline-pbr-{index}"))
        .build()
        .ok()
});

fn parallel_pixels<T: Send>(count: u32, pixel: impl Fn(u32) -> T + Sync) -> Vec<T> {
    if count >= 256 * 256
        && rayon::current_thread_index().is_none()
        && let Some(pool) = PIXEL_POOL.as_ref()
    {
        pool.install(|| (0..count).into_par_iter().map(&pixel).collect())
    } else {
        (0..count).map(pixel).collect()
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "row/column indices are bounded by the u32 image dimensions"
)]
fn rasterize(size: u32, pixel: impl Fn(u32, u32) -> Rgba<u8> + Sync) -> RgbaImage {
    if size < 256 || rayon::current_thread_index().is_some() {
        return RgbaImage::from_fn(size, size, pixel);
    }
    let Some(pool) = PIXEL_POOL.as_ref() else {
        return RgbaImage::from_fn(size, size, pixel);
    };
    pool.install(|| {
        let mut image = RgbaImage::new(size, size);
        let bytes: &mut [u8] = image.as_mut();
        bytes
            .par_chunks_mut(size as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, output) in row.chunks_exact_mut(4).enumerate() {
                    output.copy_from_slice(&pixel(x as u32, y as u32).0);
                }
            });
        image
    })
}

/// Native material templates supported by static preview baking.
pub const SUPPORTED_TEMPLATES: &[&str] = &[
    r"base\fx\_shaders\blackwall_blendable_mesh_decal_gradient.mt",
    r"base\fx\_shaders\blackwall_blendable_eye_wet.mt",
    r"base\fx\_shaders\blackwall_blendable_hair.mt",
    r"base\fx\_shaders\blackwall_blendable_eye.mt",
    r"base\fx\_shaders\blackwall_blendable_eye_gradient.mt",
    r"base\materials\eye_blendable.mt",
    r"base\materials\mesh_decal_blendable.mt",
    r"base\materials\hair_cyberspace.mt",
    r"base\fx\_shaders\blackwall_blendable_glass.mt",
    r"base\materials\skin_blendable.mt",
    r"base\materials\skin_morph.mt",
    r"base\materials\hair_proxy.mt",
    r"base\fx\_shaders\blackwall_blendable_skin.mt",
    r"base\fx\_shaders\holo_mask.mt",
    r"base\fx\shaders\animated_decal.mt",
    r"base\fx\shaders\emissive_basic_transparent.mt",
    r"base\fx\shaders\metal_base_blackbody.mt",
    r"base\materials\silverhand_overlay_blendable.mt",
    r"base\materials\cloth_mov.mt",
    r"base\materials\worldspace_grid.mt",
    r"base\materials\mesh_decal_emissive_subsurface.mt",
    r"engine\materials\metal_base_blendable.mt",
    r"base\fx\_shaders\blackwall_blendable_mesh_decal.mt",
    r"base\fx\_shaders\lightning_plasma.mt",
    r"base\fx\_shaders\multilayer_blackbody_inject.mt",
    r"base\fx\_shaders\nanowire_string_spline_deformed.mt",
    r"base\materials\silverhand_overlay.mt",
    r"base\fx\shaders\device_diode_multi_state.mt",
    r"base\fx\_shaders\blackwall_blendable_metal_base.mt",
    r"base\fx\_shaders\blackwall_blendable.mt",
    r"base\materials\glass_scope.mt",
    r"base\fx\_shaders\highlight_blocker.mt",
    r"base\fx\shaders\hologram_two_sided.mt",
    r"base\materials\frosted_glass_curtain.mt",
    r"base\materials\mesh_decal_gradientmap_recolor_emissive.mt",
    r"base\fx\shaders\neon_tubes.mt",
    r"base\fx\shaders\particles_generic_expanded.mt",
    r"engine\materials\particles_generic.remt",
    r"engine\materials\metal_base_animated_uv.mt",
    r"base\fx\shaders\blackbodyradiation.mt",
    r"base\materials\pbr_layer.remt",
    r"base\fx\_shaders\mesh_decal__blackbody.mt",
    r"base\materials\skin.mt",
    r"base\materials\hair.mt",
    r"base\materials\eye_gradient.mt",
    r"base\materials\eye.mt",
    r"base\materials\eye_shadow.mt",
    r"base\materials\mesh_decal_double_diffuse.mt",
    r"base\fx\_shaders\crystal_dome.mt",
    r"base\fx\_shaders\crystal_dome_opaque.mt",
    r"base\fx\_shaders\crystal_dome_opaque_backface.mt",
    r"base\fx\_shaders\cloak_multilayered_forward.mt",
    r"base\fx\shaders\cloak.mt",
    r"base\fx\shaders\parallaxscreen_transparent_ui.mt",
    r"base\materials\top_down_car_proxy_depth.mt",
    r"base\materials\fillable_fluid_vertex.mt",
    r"base\materials\multilayered_clear_coat.mt",
    r"base\materials\metal_base_parallax.mt",
    r"base\materials\multilayered_cable_swing.mt",
    r"base\materials\cloth_mov_multilayered.mt",
    r"base\vehicles\common\materials\vehicle_modding_destruction.mt",
    r"base\fx\_shaders\invisible.mt",
    r"base\fx\shaders\device_diode.mt",
    r"base\fx\shaders\hologram.mt",
    r"base\fx\shaders\metal_base_glitter.mt",
    r"base\fx\shaders\parallaxscreen.mt",
    r"base\fx\shaders\parallaxscreen_transparent.mt",
    r"base\fx\shaders\signages.mt",
    r"base\materials\glass.mt",
    r"base\materials\glass_cracked_edge.mt",
    r"base\materials\glass_deferred.mt",
    r"base\materials\glass_onesided.mt",
    r"base\materials\fillable_fluid.mt",
    r"base\materials\lights_interactive.mt",
    r"base\materials\metal_base_det.mt",
    r"base\materials\metal_base_det_dithered.mt",
    r"base\materials\metal_base_dithered.mt",
    r"base\materials\metal_base_gradientmap_recolor.mt",
    r"base\materials\metal_base_ui.mt",
    r"base\materials\mesh_decal.mt",
    r"base\materials\mesh_decal_emissive.mt",
    r"base\materials\mesh_decal_gradientmap_recolor.mt",
    r"base\materials\mesh_decal_parallax.mt",
    r"base\materials\multilayered_terrain.mt",
    r"base\materials\vehicle_destr_blendshape.mt",
    r"base\materials\vehicle_glass.mt",
    r"base\materials\vehicle_glass_onesided.mt",
    r"base\materials\vehicle_lights.mt",
    r"base\materials\vehicle_mesh_decal.mt",
    r"base\materials\vehicle_mesh_decal_parallax.mt",
    r"base\materials\window_parallax_interior.mt",
    r"engine\materials\metal_base.remt",
    r"engine\materials\metal_base_proxy.mt",
    r"engine\materials\multilayered.mt",
];

#[derive(Debug, Error)]
pub enum PbrError {
    #[error("could not access PBR material data: {0}")]
    Io(#[from] io::Error),
    #[error("could not decode PBR material JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("could not decode or encode PBR texture: {0}")]
    Image(#[from] ImageError),
    #[error("malformed PBR material data: {0}")]
    Malformed(String),
    #[error("unsupported material template: {0}")]
    UnsupportedTemplate(String),
    #[error("material texture was not extracted: {0}")]
    MissingTexture(PathBuf),
    #[error("{source}; native dependency resolution: {dependencies}")]
    NativeDependencies {
        #[source]
        source: Box<PbrError>,
        dependencies: String,
    },
}

/// Summary of PBR materials attached to one selected-appearance GLB.
#[derive(Clone, Debug, Default)]
pub struct BakeSummary {
    pub materials: usize,
    pub generated_textures: usize,
    pub reused_textures: usize,
}

/// Native strand-direction and vertex-color inputs for preview hair lighting.
#[derive(Clone, Debug)]
pub struct HairShading {
    /// Native tangent-space strand direction map, sampled as linear data.
    pub flow_texture: Option<PathBuf>,
    /// Authored strength of the strand direction map.
    pub flow_strength: f32,
    /// Red vertex-color channel darkens the inner hair layers.
    pub vertex_color_strength: f32,
}

/// Native mesh-decal controls retained for receiving-surface composition.
#[derive(Clone, Debug)]
pub struct DecalShading {
    /// Authored color profile for gradient-enabled double-diffuse decals.
    pub gradient_texture: Option<PathBuf>,
    /// Multiplier on the sampled color profile.
    pub gradient_intensity: f32,
    /// Authored horizontal position in the one-dimensional color profile.
    pub gradient_uv: f32,
    /// Secondary diffuse/coverage layer, sharing the primary decal UV transform.
    pub secondary_texture: Option<PathBuf>,
    /// Fallback tint when the color profile is disabled.
    pub secondary_color: [f32; 4],
    /// Coverage multiplier for the secondary layer.
    pub secondary_alpha: f32,
    /// Authored contribution of decal normals to the receiving surface.
    pub normal_alpha: f32,
    /// Authored contribution of roughness and metalness.
    pub roughness_metalness_alpha: f32,
    /// Native depth rejection threshold.
    pub depth_threshold: f32,
    /// Native normal blending mode.
    pub normal_blending_mode: f32,
    /// Whether the packed normal's alpha channel supplies native normal coverage.
    pub normal_alpha_texture: bool,
}

/// Render-oriented material parameters and baked texture paths.
#[derive(Clone, Debug)]
pub struct PbrMaterial {
    /// Material name in the native appearance.
    pub name: String,
    /// Baked base color map, interpreted as sRGB.
    pub base_color_texture: Option<PathBuf>,
    /// Linear base color multiplier.
    pub base_color_factor: [f32; 4],
    /// Texture-coordinate scale (XY) and offset (ZW), applied before sampling.
    pub uv_transform: [f32; 4],
    /// Linear packed surface map: roughness in G, metallic in B.
    pub metallic_roughness_texture: Option<PathBuf>,
    /// Metallic multiplier.
    pub metallic_factor: f32,
    /// Roughness multiplier.
    pub roughness_factor: f32,
    /// Linear tangent-space normal map.
    pub normal_texture: Option<PathBuf>,
    /// Emissive map, interpreted as sRGB.
    pub emissive_texture: Option<PathBuf>,
    /// Linear emission multiplier.
    pub emissive_factor: [f32; 3],
    /// Emission intensity.
    pub emissive_strength: f32,
    /// Coverage mode: OPAQUE, MASK or BLEND.
    pub alpha_mode: &'static str,
    /// Cutoff used by MASK coverage.
    pub alpha_cutoff: Option<f32>,
    /// Prefer depth-writing stochastic coverage for intersecting cutout surfaces.
    /// Exporters retain the standard alpha mode; native previews may hash coverage.
    pub stochastic_coverage: bool,
    /// Native strand inputs for directional hair lighting.
    pub hair: Option<HairShading>,
    /// Native mesh-decal surface blending controls.
    pub decal: Option<DecalShading>,
    /// Approximate transmission strength.
    pub transmission: f32,
    /// Whether both triangle faces are visible.
    pub double_sided: bool,
    /// Material effects approximated or omitted in the bake.
    pub preview_notes: Vec<String>,
}

impl PbrMaterial {
    #[expect(
        clippy::float_cmp,
        reason = "exact authored identity determines whether a texture-transform extension is needed"
    )]
    fn has_uv_transform(&self) -> bool {
        self.uv_transform != [1., 1., 0., 0.]
    }

    fn new(name: String) -> Self {
        Self {
            name,
            base_color_texture: None,
            base_color_factor: [1.0; 4],
            uv_transform: [1., 1., 0., 0.],
            metallic_roughness_texture: None,
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            normal_texture: None,
            emissive_texture: None,
            emissive_factor: [0.0; 3],
            emissive_strength: 1.0,
            alpha_mode: "OPAQUE",
            alpha_cutoff: None,
            stochastic_coverage: false,
            hair: None,
            decal: None,
            transmission: 0.0,
            double_sided: true,
            preview_notes: Vec::new(),
        }
    }
}

struct BakeContext<'a> {
    repository: &'a Path,
    size: u32,
    generated: usize,
    reused: usize,
}

/// Builds a visible diagnostic texture from one authored multilayer mask.
/// Magenta shows mask coverage; higher material layers are not composited here.
///
/// # Errors
/// Returns an error for absent material provenance, missing masks or image I/O.
pub fn highlight_layer(
    material: &PbrMaterial,
    manifest: &Value,
    layer: usize,
    output: &Path,
) -> Result<PbrMaterial, PbrError> {
    let record = manifest["Materials"]
        .as_array()
        .and_then(|records| records.iter().find(|r| r["Name"] == material.name))
        .ok_or_else(|| malformed("material provenance absent"))?;
    let repository = record["MaterialRepo"]
        .as_str()
        .or_else(|| manifest["MaterialRepo"].as_str())
        .ok_or_else(|| malformed("material repository absent"))?;
    let depot = record["Data"]["MultilayerMask"]
        .as_str()
        .ok_or_else(|| malformed("multilayer mask absent"))?;
    let source = depot_to_path(Path::new(repository), depot, "png");
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| malformed("mask name absent"))?;
    let mask = source
        .with_file_name(format!("{stem}_layers"))
        .join(format!("{stem}_{layer}.png"));
    if !mask.is_file() {
        return Err(PbrError::MissingTexture(mask));
    }
    if !output.is_file() {
        let mask = load_rgba(&mask, 512)?;
        let highlighted = ImageBuffer::from_fn(mask.width(), mask.height(), |x, y| {
            let coverage = f32::from(mask.get_pixel(x, y)[0]) / 255.;
            Rgba([
                (45. + 210. * coverage) as u8,
                (50. - 15. * coverage) as u8,
                (54. + 126. * coverage) as u8,
                255,
            ])
        });
        highlighted.save(output)?;
    }
    let mut result = material.clone();
    result.base_color_texture = Some(output.to_path_buf());
    result.base_color_factor = [1.; 4];
    result.normal_texture = None;
    result.metallic_roughness_texture = None;
    result.metallic_factor = 0.;
    result.roughness_factor = 1.;
    result.emissive_texture = Some(output.to_path_buf());
    result.emissive_factor = [0.6; 3];
    result.emissive_strength = 1.;
    Ok(result)
}

/// Linear colour coefficients for interactive multilayer recolouring.
/// Tiles retain native texture detail, opacity, microblend and occlusion by later layers.
pub struct ColourLayers {
    /// Native setup identity used to bind authoring regions.
    pub setup: String,
    /// Original linear colour multipliers in native layer order.
    pub colours: Vec<[f32; 3]>,
    /// Square grid of sRGB coefficient tiles, with one tile per native layer.
    pub atlas: RgbaImage,
    /// Number of tiles across the atlas.
    pub columns: u32,
}

/// Prepares reusable colour coefficients without rendering or writing game resources.
///
/// # Errors
/// Returns an error for missing provenance, native material inputs or texture I/O.
#[expect(
    clippy::cast_precision_loss,
    reason = "fixed 512px coordinates are exactly representable as f32"
)]
pub fn colour_layers(
    material: &PbrMaterial,
    manifest: &Value,
) -> Result<Option<ColourLayers>, PbrError> {
    let record = manifest["Materials"]
        .as_array()
        .and_then(|records| records.iter().find(|r| r["Name"] == material.name))
        .ok_or_else(|| malformed("material provenance absent"))?;
    let repository = Path::new(
        record["MaterialRepo"]
            .as_str()
            .or_else(|| manifest["MaterialRepo"].as_str())
            .ok_or_else(|| malformed("material repository absent"))?,
    );
    let Some(setup) = record["Data"]["MultilayerSetup"].as_str() else {
        return Ok(None);
    };
    let mask = required_resource(
        record["Data"]
            .as_object()
            .ok_or_else(|| malformed("material data absent"))?,
        "MultilayerMask",
    )?;
    let document: Value = serde_json::from_slice(&fs::read(depot_json_path(repository, setup))?)?;
    let layers = root_chunk(&document)?
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("multilayer layers absent"))?;
    if layers.is_empty() || layers.len() > 64 {
        return Err(malformed("interactive layer count must be 1..64"));
    }
    let size = DEFAULT_BAKE_SIZE;
    let masks = load_layer_masks(repository, mask, layers.len(), size)?;
    let surfaces = layers
        .iter()
        .map(|layer| load_multilayer_surface(repository, layer, size))
        .collect::<Result<Vec<_>, _>>()?;
    let count = u32::try_from(layers.len()).map_err(|_| malformed("layer count overflow"))?;
    let columns = (1..=8)
        .find(|n| n * n >= count)
        .ok_or_else(|| malformed("layer atlas overflow"))?;
    let rows = count.div_ceil(columns);
    let mut atlas = RgbaImage::new(columns * size, rows * size);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let mut remaining = 1.;
            for (index, surface) in surfaces.iter().enumerate().rev() {
                let coverage = if index == 0 {
                    1.
                } else {
                    masks
                        .get(index)
                        .map_or(0., |m| f32::from(m.get_pixel(x, y)[0]) / 255.)
                };
                let coverage = surface.microblend.as_ref().map_or(coverage, |image| {
                    let sample = sample_repeat(
                        image,
                        u * surface.microblend_uv[0] + surface.microblend_uv[1],
                        v * surface.microblend_uv[0] + surface.microblend_uv[2],
                    );
                    microblend_coverage(coverage, sample[3], surface.microblend_contrast)
                });
                let alpha = (coverage * surface.opacity).clamp(0., 1.);
                let (tx, ty) = repeat_coordinates(
                    size,
                    size,
                    u * surface.uv[0] + surface.uv[1],
                    v * surface.uv[0] + surface.uv[2],
                );
                let sample = sample_at(&surface.color, tx, ty);
                let mut pixel = [255; 4];
                for channel in 0..3 {
                    let coefficient =
                        SRGB_LINEAR[usize::from(unit_to_byte(sample[channel]))] * alpha * remaining;
                    pixel[channel] = unit_to_byte(linear_to_srgb(coefficient));
                }
                let tile = u32::try_from(index).map_err(|_| malformed("layer index overflow"))?;
                atlas.put_pixel(
                    (tile % columns) * size + x,
                    (tile / columns) * size + y,
                    Rgba(pixel),
                );
                remaining *= 1. - alpha;
            }
        }
    }
    Ok(Some(ColourLayers {
        setup: setup.into(),
        colours: surfaces.iter().map(|s| s.color_scale).collect(),
        atlas,
        columns,
    }))
}

/// Bakes the sidecar's selected appearance and attaches standard glTF PBR materials.
///
/// The source GLB geometry is preserved. Generated textures are content-addressed
/// beneath the shared material repository and reused across meshes.
///
/// # Errors
///
/// Returns [`PbrError`] when the sidecar is malformed, a referenced dependency
/// is missing, a template is unsupported, or the GLB cannot be rewritten.
pub fn bake_sidecar_into_glb(
    sidecar: &Path,
    glb: &Path,
    appearance: &str,
    size: Option<u32>,
) -> Result<BakeSummary, PbrError> {
    let (geometry, _) = decode_glb(&fs::read(glb)?)?;
    let chunks = geometry
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("GLB has no meshes"))?;
    let indices: Vec<_> = chunks
        .iter()
        .enumerate()
        .map(|(index, mesh)| {
            mesh.get("name")
                .and_then(Value::as_str)
                .and_then(|name| name.strip_prefix("submesh_"))
                .and_then(|name| name.split('_').next())
                .and_then(|name| name.parse::<usize>().ok())
                .unwrap_or(index)
        })
        .collect();
    let baked = bake_appearance(sidecar, appearance, &indices, size)?;
    attach_materials(glb, &baked.chunk_materials, &baked.materials)?;
    Ok(baked.summary)
}

/// Materials and the original chunk-to-material appearance table.
#[derive(Clone, Debug)]
pub struct BakedAppearance {
    /// Native material names, indexed by render chunk using native wrapping rules.
    pub chunk_materials: Vec<String>,
    /// Unique baked materials needed by the selected chunks.
    pub materials: Vec<PbrMaterial>,
    /// Texture generation and reuse counts.
    pub summary: BakeSummary,
}

/// Bakes materials for selected native render chunks without a geometry export.
///
/// Texture results are cached beneath the sidecar's material repository. Chunk
/// indices must be the original indices, including after LOD filtering.
///
/// # Errors
///
/// Returns [`PbrError`] for missing appearances, malformed sidecars, unsupported
/// material templates, missing textures or output errors.
pub fn bake_appearance(
    sidecar: &Path,
    appearance: &str,
    chunk_indices: &[usize],
    size: Option<u32>,
) -> Result<BakedAppearance, PbrError> {
    let document: Value = serde_json::from_slice(&fs::read(sidecar)?)?;
    let repository = PathBuf::from(
        document
            .get("MaterialRepo")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("sidecar has no MaterialRepo"))?,
    );
    let appearance_materials = document
        .get("Appearances")
        .and_then(|value| value.get(appearance))
        .and_then(Value::as_array)
        .ok_or_else(|| malformed(format!("appearance {appearance:?} was not found")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| malformed("appearance material name is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let records = document
        .get("Materials")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("sidecar has no Materials array"))?;
    let by_name: HashMap<_, _> = records
        .iter()
        .filter_map(|record| {
            record
                .get("Name")
                .and_then(Value::as_str)
                .map(|name| (name.to_owned(), record))
        })
        .collect();
    if appearance_materials.is_empty() || chunk_indices.is_empty() {
        return Err(malformed("appearance and render chunks must be nonempty"));
    }
    let selected: HashSet<_> = chunk_indices
        .iter()
        .map(|&chunk| material_for_chunk(&appearance_materials, chunk).to_owned())
        .collect();
    let mut context = BakeContext {
        repository: &repository,
        size: size.unwrap_or(DEFAULT_BAKE_SIZE).max(1),
        generated: 0,
        reused: 0,
    };
    let mut materials = Vec::with_capacity(selected.len());
    for name in appearance_materials
        .iter()
        .filter(|name| selected.contains(name.as_str()))
    {
        if materials
            .iter()
            .any(|material: &PbrMaterial| material.name == *name)
        {
            continue;
        }
        let record = by_name
            .get(name)
            .ok_or_else(|| malformed(format!("material {name:?} has no sidecar record")))?;
        let material_repository = record
            .get("MaterialRepo")
            .and_then(Value::as_str)
            .map(PathBuf::from);
        let mut material_context = BakeContext {
            repository: material_repository.as_deref().unwrap_or(&repository),
            size: context.size,
            generated: 0,
            reused: 0,
        };
        let mut material = bake_material(record, &mut material_context).map_err(|error| {
            let dependencies = document
                .get("DependencyErrors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            if matches!(&error, PbrError::MissingTexture(_) | PbrError::Io(_))
                && !dependencies.is_empty()
            {
                PbrError::NativeDependencies {
                    source: Box::new(error),
                    dependencies: dependencies.join("; "),
                }
            } else {
                error
            }
        })?;
        context.generated += material_context.generated;
        context.reused += material_context.reused;
        if let Some(fallbacks) = record.get("ResourceFallbacks").and_then(Value::as_array) {
            for fallback in fallbacks {
                material
                    .preview_notes
                    .push(format!("Native texture fallback: {fallback}"));
            }
        }
        materials.push(material);
    }
    // Hair cards and their companion cap must occlude each other at strand depth.
    // Keep unrelated decal appearances on their usual alpha-blending path.
    if materials.iter().any(|material| material.hair.is_some()) {
        for material in &mut materials {
            if material.alpha_mode == "BLEND" {
                material.stochastic_coverage = true;
            }
        }
    }
    let summary = BakeSummary {
        materials: materials.len(),
        generated_textures: context.generated,
        reused_textures: context.reused,
    };
    Ok(BakedAppearance {
        chunk_materials: appearance_materials,
        materials,
        summary,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "explicit dispatch keeps native material families together"
)]
fn bake_material(record: &Value, context: &mut BakeContext<'_>) -> Result<PbrMaterial, PbrError> {
    let name = required_string(record, "Name")?.to_owned();
    let template = required_string(record, "MaterialTemplate")?
        .replace('/', "\\")
        .to_lowercase();
    if !SUPPORTED_TEMPLATES.contains(&template.as_str()) && !environment::supports(&template) {
        return Err(PbrError::UnsupportedTemplate(template));
    }
    let data = record
        .get("Data")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("material Data is not an object"))?;
    let enable_mask = record
        .get("EnableMask")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    match template.as_str() {
        r"base\fx\shaders\metal_base_blackbody.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            material.preview_notes.push("Thermal metal retains authored base color, metalness, roughness and normal maps in its inactive static state; runtime heat distribution, temperature curves and thermal emission are not supplied".into());
            Ok(material)
        }
        r"base\materials\metal_base_det.mt" | r"base\materials\metal_base_det_dithered.mt" => {
            let mut material = bake_metal_base(
                name,
                record,
                data,
                context,
                enable_mask || template.ends_with("_dithered.mt"),
            )?;
            let solid_editor_map = |key: &str, leaf: &str| {
                data.get(key).and_then(resource_string).is_some_and(|path| {
                    path.replace('/', "\\")
                        .eq_ignore_ascii_case(&format!(r"engine\textures\editor\{leaf}.xbm"))
                })
            };
            if solid_editor_map("BaseColor", "white")
                && solid_editor_map("Normal", "normal")
                && solid_editor_map("Roughness", "grey")
                && solid_editor_map("Metalness", "white")
            {
                // Chain-link and similar cutouts use constant base maps. Keep
                // the detail image and native UV scale directly, preserving
                // alpha/normal mip filtering instead of aliasing a tiled bake.
                material.base_color_texture =
                    optional_texture(context.repository, data, "DetailColor")?;
                material.normal_texture =
                    optional_texture(context.repository, data, "DetailNormal")?;
                material.uv_transform = [
                    scalar(data, "DetailU", 10.),
                    scalar(data, "DetailV", 5.),
                    0.,
                    0.,
                ];
                return Ok(material);
            }
            let base = material
                .base_color_texture
                .as_deref()
                .map(|path| load_rgba(path, context.size))
                .transpose()?;
            let detail = optional_texture(context.repository, data, "DetailColor")?
                .as_deref()
                .map(|path| load_rgba(path, context.size))
                .transpose()?;
            let normal = material
                .normal_texture
                .as_deref()
                .map(|path| load_rgba(path, context.size))
                .transpose()?;
            let detail_normal = optional_texture(context.repository, data, "DetailNormal")?
                .as_deref()
                .map(|path| load_rgba(path, context.size))
                .transpose()?;
            let tile = [scalar(data, "DetailU", 10.), scalar(data, "DetailV", 5.)];
            material.base_color_texture = Some(character::bake_map(
                record,
                "metal-detail-color.png",
                context,
                |size| {
                    RgbaImage::from_fn(size, size, |x, y| {
                        let uv = [
                            (x as f32 + 0.5) / size as f32,
                            (y as f32 + 0.5) / size as f32,
                        ];
                        let base = base
                            .as_ref()
                            .map_or([1.; 4], |map| sample_repeat(map, uv[0], uv[1]));
                        let detail = detail.as_ref().map_or([1.; 4], |map| {
                            sample_repeat(map, uv[0] * tile[0], uv[1] * tile[1])
                        });
                        Rgba(std::array::from_fn(|i| {
                            (base[i] * detail[i] * 255.).clamp(0., 255.).round() as u8
                        }))
                    })
                },
            )?);
            if detail_normal.is_some() {
                material.normal_texture = Some(character::bake_map(
                    record,
                    "metal-detail-normal.png",
                    context,
                    |size| {
                        RgbaImage::from_fn(size, size, |x, y| {
                            let uv = [
                                (x as f32 + 0.5) / size as f32,
                                (y as f32 + 0.5) / size as f32,
                            ];
                            let base = normal
                                .as_ref()
                                .map_or([0.5, 0.5, 1., 1.], |map| sample_repeat(map, uv[0], uv[1]));
                            let detail = detail_normal.as_ref().map_or([0.5, 0.5, 1., 1.], |map| {
                                sample_repeat(map, uv[0] * tile[0], uv[1] * tile[1])
                            });
                            let combined = glam::Vec3::new(
                                (base[0] + detail[0] - 1.) * 2.,
                                (base[1] + detail[1] - 1.) * 2.,
                                (base[2] * 2. - 1.) * (detail[2] * 2. - 1.),
                            )
                            .normalize_or_zero();
                            Rgba([
                                ((combined.x * 0.5 + 0.5) * 255.).clamp(0., 255.).round() as u8,
                                ((combined.y * 0.5 + 0.5) * 255.).clamp(0., 255.).round() as u8,
                                ((combined.z * 0.5 + 0.5) * 255.).clamp(0., 255.).round() as u8,
                                255,
                            ])
                        })
                    },
                )?);
            }
            Ok(material)
        }
        r"base\materials\cloth_mov.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            let tile = scalar(data, "LayerTile", 1.);
            material.uv_transform = [tile, tile, 0., 0.];
            material.preview_notes.push("Moving cloth retains its authored bind geometry, base color, normals, surface maps and emission; runtime vertex-texture animation is not evaluated".into());
            Ok(material)
        }
        r"base\materials\worldspace_grid.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            let scale = scalar(data, "GridScale", 1.);
            material.uv_transform = [scale, scale, 0., 0.];
            if scalar(data, "EnableWorldSpace", 0.) > 0. {
                material.preview_notes.push("Grid preview uses its authored checker/color/scale on mesh UVs; runtime world-space projection is approximated".into());
            }
            Ok(material)
        }
        r"base\materials\mesh_decal_emissive_subsurface.mt" => {
            let primary = optional_texture(context.repository, data, "EmissiveMask")?
                .as_deref()
                .map(|p| load_rgba(p, context.size))
                .transpose()?;
            let secondary = optional_texture(context.repository, data, "SecondaryMask")?
                .as_deref()
                .map(|p| load_rgba(p, context.size))
                .transpose()?;
            let weights = data
                .get("EmissiveMaskChannel")
                .map_or([1., 0., 0., 0.], |value| {
                    [
                        scalar_object(value, "X", 1.),
                        scalar_object(value, "Y", 0.),
                        scalar_object(value, "Z", 0.),
                        scalar_object(value, "W", 0.),
                    ]
                });
            let texture = character::bake_map(
                record,
                "emissive-subsurface-coverage.png",
                context,
                |size| {
                    RgbaImage::from_fn(size, size, |x, y| {
                        let u = (x as f32 + 0.5) / size as f32;
                        let v = (y as f32 + 0.5) / size as f32;
                        let channels = primary
                            .as_ref()
                            .map_or([1.; 4], |image| sample_repeat(image, u, v));
                        let mask = channels
                            .iter()
                            .zip(weights)
                            .map(|(channel, weight)| channel * weight)
                            .sum::<f32>();
                        let secondary = secondary
                            .as_ref()
                            .map_or(1., |image| sample_repeat(image, u, v)[0]);
                        Rgba([
                            255,
                            255,
                            255,
                            (mask.mul_add(secondary, 0.).clamp(0., 1.) * 255.).round() as u8,
                        ])
                    })
                },
            )?;
            let mut material = PbrMaterial::new(name);
            material.base_color_texture = Some(texture);
            material.base_color_factor = [0., 0., 0., 1.];
            material.emissive_factor = gamma_color(data.get("EmissiveColor"), [1., 0., 0., 1.])
                [..3]
                .try_into()
                .expect("RGB");
            let native_ev = scalar(data, "EmissiveEV", 2.076_205);
            // These authored EV 30 surfaces rely on game camera exposure. An
            // emissive-only catalogue image has no receiving skin or calibrated
            // game camera. Normalize its exposure so the authored hue and mask
            // remain visible instead of clipping every channel to white.
            material.emissive_strength = 1.;
            let cutoff = scalar(data, "AlphaThreshold", 0.);
            material.alpha_mode = if cutoff > 0. { "MASK" } else { "BLEND" };
            material.alpha_cutoff = (cutoff > 0.).then_some(cutoff);
            material.preview_notes.push(format!("Emissive subsurface decal retains authored channel/secondary coverage masks and color; native EV {native_ev} is exposure-normalized for its isolated studio preview. Transmission through underlying skin is approximated by surface emission"));
            Ok(material)
        }
        r"engine\materials\metal_base_blendable.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            let tile = scalar(data, "LayerTile", 1.);
            material.uv_transform = [tile, tile, 0., 0.];
            material.preview_notes.push("Blendable metal uses its authored static surface; runtime vector-field glitches and transitions are not supplied".into());
            Ok(material)
        }
        r"base\fx\_shaders\blackwall_blendable_mesh_decal.mt" => {
            let mut material = bake_decal(name, record, data, context)?;
            material.preview_notes.push("Blackwall decal retains its authored inactive surface and coverage; animated activation and displacement are not supplied".into());
            Ok(material)
        }
        r"base\fx\_shaders\multilayer_blackbody_inject.mt" => {
            let mut material = bake_multilayered(name, record, data, context)?;
            material.emissive_texture =
                optional_texture(context.repository, data, "DamageTypeRGBMask")?;
            material.emissive_factor = [1.; 3];
            material.emissive_strength = scalar(data, "Emissive", 0.).max(0.).exp2();
            material.preview_notes.push("Thermal multilayer keeps native surface layers and damage-channel emission; runtime temperature, noise and HSV thermal response are approximated".into());
            Ok(material)
        }
        r"base\materials\silverhand_overlay.mt"
        | r"base\materials\silverhand_overlay_blendable.mt" => {
            let mut material = bake_multilayered(name, record, data, context)?;
            material.alpha_mode = "BLEND";
            material.base_color_factor[3] =
                (1. - scalar(data, "BayerIntensity", 0.5) * 0.5).clamp(0., 1.);
            material.stochastic_coverage = true;
            material.preview_notes.push("Silverhand overlay uses its authored multilayer surface and static stochastic coverage; animated vector-field glitches and view-dependent Fresnel masking are approximated".into());
            Ok(material)
        }
        r"base\fx\_shaders\lightning_plasma.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_texture =
                optional_texture(context.repository, data, "TemperatureTexture")?;
            material.emissive_texture = material.base_color_texture.clone();
            material.base_color_factor = gamma_color(data.get("Tint"), [1.; 4]);
            material.emissive_factor = material.base_color_factor[..3]
                .try_into()
                .expect("three color channels");
            material.base_color_factor[3] *= scalar(data, "maxAlpha", 1.).clamp(0., 1.);
            material.alpha_mode = "BLEND";
            material.stochastic_coverage = true;
            material.preview_notes.push("Plasma uses authored temperature texture, tint and coverage in its static geometry; animated displacement and thermal color response are approximated".into());
            Ok(material)
        }
        r"base\fx\_shaders\nanowire_string_spline_deformed.mt" => {
            let mut material = PbrMaterial::new(name);
            material.base_color_factor = gamma_color(data.get("MainColor"), [1.; 4]);
            material.normal_texture = optional_texture(context.repository, data, "NormalMap")?;
            material.metallic_factor = scalar(data, "Metalness", 0.).clamp(0., 1.);
            material.roughness_factor = scalar(data, "Roughness", 0.5).clamp(0., 1.);
            material.emissive_factor = material.base_color_factor[..3]
                .try_into()
                .expect("three color channels");
            material.emissive_strength = scalar(data, "EmissiveEV", 0.).clamp(-32., 32.).exp2()
                * scalar(data, "MinimumEmissive", 0.).max(0.);
            material.preview_notes.push("Monowire uses authored static cable geometry, color, normal, surface and minimum emission; runtime spline extension, noise and thermal color response are not simulated".into());
            Ok(material)
        }
        r"base\fx\_shaders\blackwall_blendable.mt" => {
            let mut material = bake_multilayered(name, record, data, context)?;
            material.preview_notes.push("Blackwall blendable retains the authored multilayer surface in its inactive static state; runtime activation, animated displacement and dither transitions are not evaluated".into());
            Ok(material)
        }
        r"base\fx\_shaders\blackwall_blendable_metal_base.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            let tile = scalar(data, "LayerTile", 1.);
            material.uv_transform = [tile, tile, 0., 0.];
            material.preview_notes.push("Blackwall material retains its authored static metal surface; runtime Blackwall activation is not supplied".into());
            Ok(material)
        }
        r"base\fx\shaders\device_diode_multi_state.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            let selector = (scalar(data, "EmissiveColorSelector", 1.).round() - 1.).clamp(0., 3.);
            let key = if selector < 1. {
                "EmissiveColor1"
            } else if selector < 2. {
                "EmissiveColor2"
            } else if selector < 3. {
                "EmissiveColor3"
            } else {
                "EmissiveColor4"
            };
            material.emissive_factor = color(data.get(key), [1.; 4])[..3]
                .try_into()
                .expect("three color channels");
            material.preview_notes.push(format!("Multi-state diode uses authored {key}; runtime state changes and blinking are not simulated"));
            Ok(material)
        }
        r"base\fx\_shaders\blackwall_blendable_glass.mt" => {
            let mut material = bake_glass(name, record, data, context)?;
            material.preview_notes.push("Blackwall glass retains the authored static tint, coverage, normal and roughness; animated activation and displacement are not supplied".into());
            Ok(material)
        }
        r"base\materials\glass_scope.mt" => {
            let mut material = bake_glass(name, record, data, context)?;
            material.base_color_texture = optional_texture(context.repository, data, "GlassTint")?;
            material.base_color_factor = gamma_color(data.get("GlassTintMultiplier"), [1.; 4]);
            material.base_color_factor[3] = scalar(data, "MaskOpacity", 0.2).clamp(0.02, 1.);
            material.roughness_factor = scalar(data, "LensRoughness", 0.05).clamp(0., 1.);
            material.metallic_factor = scalar(data, "LensMetalness", 0.).clamp(0., 1.);
            if let Some(uv) = data.get("UvTilingOffset") {
                material.uv_transform = [
                    scalar_object(uv, "X", 1.),
                    scalar_object(uv, "Y", 1.),
                    scalar_object(uv, "Z", 0.),
                    scalar_object(uv, "W", 0.),
                ];
                if material.uv_transform[0].abs() < f32::EPSILON
                    && material.uv_transform[1].abs() < f32::EPSILON
                {
                    material.uv_transform[0] = 1.;
                    material.uv_transform[1] = 1.;
                }
            }
            material.preview_notes.push("Scope glass retains native tint, texture, normals, surface maps and lens controls; a live scope camera and optical distortion are not evaluated".into());
            Ok(material)
        }
        r"base\materials\mesh_decal_gradientmap_recolor_emissive.mt" => {
            let mut material = character::gradient_decal(name, record, data, context)?;
            material.emissive_texture = material.base_color_texture.clone();
            material.emissive_factor = [1.; 3];
            material.emissive_strength = scalar(data, "EmissiveEV", 1.).max(0.);
            Ok(material)
        }
        r"base\fx\shaders\neon_tubes.mt" => bake_neon_tubes(name, record, data, context),
        r"base\fx\shaders\particles_generic_expanded.mt"
        | r"engine\materials\particles_generic.remt" => {
            bake_particle_fx(name, record, data, context)
        }
        r"engine\materials\metal_base_animated_uv.mt" => {
            let mut material = bake_metal_base(name, record, data, context, enable_mask)?;
            material.preview_notes.push("Animated metal UVs are sampled at time zero; authored base color, surface maps, normals and emission are preserved".into());
            Ok(material)
        }
        r"base\fx\shaders\blackbodyradiation.mt" => bake_blackbody_fx(name, record, data, context),
        r"base\materials\pbr_layer.remt" => {
            bake_pbr_layer(name, record, data, context, enable_mask)
        }
        r"base\fx\_shaders\mesh_decal__blackbody.mt" => {
            if scalar(data, "UseFloatParam", 0.) <= 0. {
                return Err(PbrError::UnsupportedTemplate(format!(
                    "{template} with authored static heat"
                )));
            }
            let mut material = bake_invisible(name);
            material.preview_notes.push("Runtime-driven blackbody heat overlay is transparent in the parked preview; no engine heat parameter is supplied".into());
            Ok(material)
        }
        r"base\materials\skin.mt" => character::skin(name, record, data, context),
        r"base\materials\skin_blendable.mt" | r"base\fx\_shaders\blackwall_blendable_skin.mt" => {
            let mut material = character::skin(name, record, data, context)?;
            material.preview_notes.push("Blendable skin retains its authored static skin surface; runtime glitch, fade and Blackwall activation are not supplied".into());
            Ok(material)
        }
        r"base\materials\skin_morph.mt" => character::skin_morph(name, record, data, context),
        r"base\materials\hair_proxy.mt" => character::hair_proxy(name, record, data, context),
        r"base\materials\mesh_decal_blendable.mt" | r"base\fx\shaders\animated_decal.mt" => {
            let mut material = bake_decal(name, record, data, context)?;
            material.uv_transform = [
                scalar(data, "UVScaleX", 1.) / scalar(data, "AnimationFramesWidth", 1.).max(1.),
                scalar(data, "UVScaleY", 1.) / scalar(data, "AnimationFramesHeight", 1.).max(1.),
                scalar(data, "UVOffsetX", 0.),
                scalar(data, "UVOffsetY", 0.),
            ];
            material.preview_notes.push("Animated/blendable decal displays the first authored atlas frame with native UV scale/offset, surface maps and coverage; runtime frame/reveal animation, fade and glitches are frozen".into());
            Ok(material)
        }
        r"base\fx\shaders\emissive_basic_transparent.mt" => {
            character::transparent_emission(name, record, data, context)
        }
        r"base\fx\_shaders\holo_mask.mt" => character::holo_mask(name, record, data, context),
        r"base\materials\hair.mt" => character::hair(name, record, data, context),
        r"base\materials\hair_cyberspace.mt" | r"base\fx\_shaders\blackwall_blendable_hair.mt" => {
            let mut material = character::hair(name, record, data, context)?;
            material.preview_notes.push("Blendable/cyberspace hair retains authored strand profile, coverage, flow and surface response; animated vector-field displacement, Fresnel glow and glitch transitions are not supplied".into());
            Ok(material)
        }
        r"base\materials\mesh_decal_gradientmap_recolor.mt"
        | r"base\fx\_shaders\blackwall_blendable_mesh_decal_gradient.mt" => {
            character::gradient_decal(name, record, data, context)
        }
        r"base\materials\eye_gradient.mt" => character::eye(name, record, data, context),
        r"base\materials\eye.mt" => character::eye_plain(name, record, data, context),
        r"base\fx\_shaders\blackwall_blendable_eye.mt" | r"base\materials\eye_blendable.mt" => {
            let mut material = character::eye_plain(name, record, data, context)?;
            material.preview_notes.push("Blendable eye uses its authored static surface; runtime glitch, fade and Blackwall activation are not supplied".into());
            Ok(material)
        }
        r"base\fx\_shaders\blackwall_blendable_eye_gradient.mt" => {
            let mut material = character::eye(name, record, data, context)?;
            material.preview_notes.push("Blackwall eye retains its authored iris gradient and static surface; runtime activation is not supplied".into());
            Ok(material)
        }
        r"base\materials\eye_shadow.mt" | r"base\fx\_shaders\blackwall_blendable_eye_wet.mt" => {
            character::eye_shadow(name, record, data, context)
        }
        r"base\materials\mesh_decal_double_diffuse.mt" => {
            let mut material = bake_decal(name, record, data, context)?;
            let decal = material.decal.as_mut().expect("decal shading was prepared");
            if scalar(data, "UseGradientMap", 0.) > 0. {
                decal.gradient_texture = optional_texture(context.repository, data, "GradientMap")?;
                decal.gradient_intensity = scalar(data, "GradientMapIntensity", 1.).max(0.);
                decal.gradient_uv = scalar(data, "GradientMapUV", 1.);
            }
            decal.secondary_texture =
                optional_texture(context.repository, data, "SecondaryDiffuseAlpha")?;
            decal.secondary_color = gamma_color(data.get("SecondaryDiffuseColor"), [1.; 4]);
            decal.secondary_alpha =
                scalar(data, "SecondaryDiffuseAlphaIntensity", 1.).clamp(0., 1.);
            material.preview_notes.push("Double-diffuse decal uses the authored gradient lookup position and intensity, with separate secondary diffuse coverage".into());
            Ok(material)
        }
        r"engine\materials\multilayered.mt"
        | r"base\fx\_shaders\cloak_multilayered_forward.mt"
        | r"base\materials\multilayered_clear_coat.mt"
        | r"base\materials\multilayered_cable_swing.mt"
        | r"base\materials\cloth_mov_multilayered.mt"
        | r"base\vehicles\common\materials\vehicle_modding_destruction.mt"
        | r"base\materials\multilayered_terrain.mt"
        | r"base\materials\vehicle_destr_blendshape.mt" => {
            bake_multilayered(name, record, data, context)
        }
        r"engine\materials\metal_base.remt"
        | r"base\materials\metal_base_parallax.mt"
        | r"engine\materials\metal_base_proxy.mt"
        | r"base\fx\shaders\device_diode.mt"
        | r"base\fx\shaders\metal_base_glitter.mt"
        | r"base\materials\lights_interactive.mt"
        | r"base\materials\vehicle_lights.mt"
        | r"base\materials\metal_base_gradientmap_recolor.mt"
        | r"base\materials\metal_base_ui.mt" => {
            bake_metal_base(name, record, data, context, enable_mask)
        }
        r"base\materials\metal_base_dithered.mt" => {
            bake_metal_base(name, record, data, context, true)
        }
        r"base\materials\mesh_decal.mt"
        | r"base\materials\mesh_decal_parallax.mt"
        | r"base\materials\vehicle_mesh_decal_parallax.mt"
        | r"base\materials\vehicle_mesh_decal.mt" => bake_decal(name, record, data, context),
        r"base\materials\mesh_decal_emissive.mt" => bake_emissive_decal(name, data, context),
        r"base\materials\vehicle_glass.mt" | r"base\materials\vehicle_glass_onesided.mt" => {
            let mut material = bake_glass(name, record, data, context)?;
            // Static studio approximation: alpha coverage represents clear
            // transmission without reproducing runtime damage/refraction.
            material.base_color_factor[3] *= 0.18;
            material.roughness_factor *= 0.12;
            material.preview_notes.push("Static vehicle glazing approximates authored tint and coverage; runtime refraction and glass damage are not evaluated".into());
            Ok(material)
        }
        r"base\materials\fillable_fluid.mt"
        | r"base\materials\fillable_fluid_vertex.mt"
        | r"base\materials\glass.mt"
        | r"base\materials\glass_cracked_edge.mt"
        | r"base\materials\glass_deferred.mt"
        | r"base\materials\glass_onesided.mt"
        | r"base\materials\frosted_glass_curtain.mt" => bake_glass(name, record, data, context),
        r"base\fx\shaders\parallaxscreen.mt" | r"base\fx\shaders\parallaxscreen_transparent.mt" => {
            bake_screen(name, data, context)
        }
        r"base\fx\shaders\parallaxscreen_transparent_ui.mt" => {
            // UI pixels are supplied by the running game, not an authored
            // texture. An unbound UI layer is transparent in a parked preview.
            if data
                .get("ParalaxTexture")
                .and_then(resource_string)
                .is_some()
            {
                bake_screen(name, data, context)
            } else {
                Ok(bake_invisible(name))
            }
        }
        r"base\fx\_shaders\crystal_dome.mt" => {
            let mut material = PbrMaterial::new(name);
            // Static glazing approximation, matching the vehicle importer's
            // established preview. Runtime scanlines are a separate overlay.
            material.base_color_factor = [0.015, 0.025, 0.035, 1.0];
            material.roughness_factor = 0.12;
            material.metallic_factor = 0.35;
            material.transmission = 0.1;
            Ok(material)
        }
        r"base\fx\_shaders\crystal_dome_opaque.mt"
        | r"base\fx\_shaders\crystal_dome_opaque_backface.mt"
        | r"base\fx\shaders\cloak.mt"
        | r"base\materials\top_down_car_proxy_depth.mt"
        | r"base\fx\_shaders\invisible.mt"
        | r"base\fx\_shaders\highlight_blocker.mt" => Ok(bake_invisible(name)),
        r"base\fx\shaders\signages.mt" => bake_signage(name, data, context),
        r"base\fx\shaders\hologram.mt" | r"base\fx\shaders\hologram_two_sided.mt" => {
            bake_hologram(name, data, context)
        }
        r"base\materials\window_parallax_interior.mt" => bake_window_parallax(name, data, context),
        _ => environment::bake(name, &template, record, data, context, enable_mask),
    }
}

fn bake_particle_fx(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let output = cache_directory(context.repository, record, context.size)?.join("particle-fx.png");
    let lock = bake_lock(&output);
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if output.is_file() {
        context.reused += 1;
    } else {
        let diffuse = optional_texture(context.repository, data, "Diffuse")?
            .map(|path| load_rgba(&path, context.size))
            .transpose()?;
        let mask = optional_texture(context.repository, data, "AlphaMask")?
            .map(|path| load_rgba(&path, context.size))
            .transpose()?;
        let color = gamma_color(data.get("Color"), [1.; 4]);
        let columns = scalar(data, "SubUVWidth", 1.).max(1.);
        let rows = scalar(data, "SubUVHeight", 1.).max(1.);
        let scale = data.get("TexCoordScale");
        let scale_u = scale.map_or(1., |v| scalar_object(v, "X", 1.));
        let scale_v = scale.map_or(1., |v| scalar_object(v, "Y", 1.));
        let mut image = RgbaImage::new(context.size, context.size);
        for y in 0..context.size {
            for x in 0..context.size {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "validated small texture dimensions"
                )]
                let (u, v) = (
                    (x as f32 + 0.5) / context.size as f32,
                    (y as f32 + 0.5) / context.size as f32,
                );
                let (u, v) = if scalar(data, "FlipUVby90deg", 0.) > 0. {
                    (v, 1. - u)
                } else {
                    (u, v)
                };
                let sample = diffuse.as_ref().map_or([1.; 4], |image| {
                    sample_repeat(image, u * scale_u / columns, v * scale_v / rows)
                });
                let coverage = if scalar(data, "UseAlphaMask", 1.) > 0. {
                    mask.as_ref()
                        .map_or(1., |image| sample_repeat(image, u, v)[0])
                } else {
                    1.
                };
                let one_channel = scalar(data, "UseOneChannel", 0.) > 0.;
                let rgb: [f32; 3] = std::array::from_fn(|channel| {
                    let input = if one_channel {
                        sample[0]
                    } else {
                        sample[channel]
                    };
                    lerp(
                        input,
                        sample[0] * 0.2126 + sample[1] * 0.7152 + sample[2] * 0.0722,
                        scalar(data, "Desaturate", 0.).clamp(0., 1.),
                    )
                    .max(0.)
                    .powf(scalar(data, "ColorPower", 1.).max(0.001))
                        * color[channel]
                });
                let alpha = sample[3] * color[3] * coverage * scalar(data, "AlphaGlobal", 1.);
                image.put_pixel(
                    x,
                    y,
                    Rgba([
                        unit_to_byte(linear_to_srgb(rgb[0])),
                        unit_to_byte(linear_to_srgb(rgb[1])),
                        unit_to_byte(linear_to_srgb(rgb[2])),
                        unit_to_byte(alpha),
                    ]),
                );
            }
        }
        fs::create_dir_all(output.parent().expect("cache output has a parent"))?;
        save_png(&image, &output)?;
        context.generated += 1;
    }
    let mut material = PbrMaterial::new(name);
    material.base_color_factor = [0., 0., 0., 1.];
    material.base_color_texture = Some(output.clone());
    material.emissive_texture = Some(output);
    material.emissive_factor = [1.; 3];
    material.emissive_strength = scalar(data, "ColorMultiplier", 1.).max(0.)
        * scalar(data, "EVCompensation", 0.).clamp(-32., 32.).exp2();
    material.alpha_mode = "BLEND";
    material.preview_notes.push("Static particle FX samples the first authored atlas frame, diffuse color, tint, intensity and alpha mask. Runtime scrolling, distortion, depth softening, view-dependent Fresnel and additive compositing are approximated by a blended emissive surface".into());
    Ok(material)
}

fn bake_neon_tubes(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_factor = [0., 0., 0., 1.];
    material.emissive_factor = gamma_color(data.get("color"), [1.; 4])[..3]
        .try_into()
        .expect("three color channels");
    material.emissive_strength = scalar(data, "EmissiveEV", 6.).clamp(-32., 32.).exp2();
    material.roughness_factor = scalar(data, "RoughnessBias", 0.01).clamp(0., 1.);
    material.uv_transform = [
        scalar(data, "TextureU", 1.),
        scalar(data, "TextureV", 1.),
        0.,
        0.,
    ];
    let intensity = scalar(data, "TextureIntensity", 0.).clamp(0., 1.);
    if intensity > 0.
        && let Some(path) = optional_texture(context.repository, data, "tex1")?
    {
        let output =
            cache_directory(context.repository, record, context.size)?.join("neon-emission.png");
        let lock = bake_lock(&output);
        let _guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if output.is_file() {
            context.reused += 1;
        } else {
            let mut texture = load_rgba(&path, context.size)?;
            for pixel in Arc::make_mut(&mut texture).pixels_mut() {
                for channel in &mut pixel.0[..3] {
                    *channel = unit_to_byte(linear_to_srgb(lerp(
                        1.,
                        srgb_to_linear(f32::from(*channel) / 255.),
                        intensity,
                    )));
                }
            }
            fs::create_dir_all(output.parent().expect("cache output has a parent"))?;
            save_png(&texture, &output)?;
            context.generated += 1;
        }
        material.emissive_texture = Some(output);
    }
    material.preview_notes.push("Neon tubes use authored color, exposure, texture intensity and UV scale; runtime flicker, ray-traced illumination and view-dependent edge modulation are not simulated".into());
    Ok(material)
}

fn bake_blackbody_fx(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let output =
        cache_directory(context.repository, record, context.size)?.join("blackbody-fx.png");
    let lock = bake_lock(&output);
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if output.is_file() {
        context.reused += 1;
    } else {
        let scatter = optional_texture(context.repository, data, "FireScatterAlpha")?
            .map(|path| load_rgba(&path, context.size))
            .transpose()?;
        let mask = optional_texture(context.repository, data, "MaskTexture")?
            .map(|path| load_rgba(&path, context.size))
            .transpose()?;
        let light = gamma_color(data.get("LightSmoke"), [1.; 4]);
        let dark = gamma_color(data.get("DarkSmoke"), [0., 0., 0., 1.]);
        let tiling = data.get("MaskTilingAndSpeed");
        let tile_u = tiling.map_or(1., |v| scalar_object(v, "X", 1.));
        let tile_v = tiling.map_or(1., |v| scalar_object(v, "Y", 1.));
        let columns = scalar(data, "subUVWidth", 1.).max(1.);
        let rows = scalar(data, "subUVHeight", 1.).max(1.);
        let exponent = scalar(data, "AlphaExponent", 1.).max(0.001);
        let max_alpha = scalar(data, "maxAlpha", 1.).clamp(0., 1.);
        let no_texture_alpha = scalar(data, "NoAlphaOnTexture", 0.) > 0.;
        let mut image = RgbaImage::new(context.size, context.size);
        for y in 0..context.size {
            for x in 0..context.size {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "validated small texture dimensions"
                )]
                let (u, v) = (
                    (x as f32 + 0.5) / context.size as f32,
                    (y as f32 + 0.5) / context.size as f32,
                );
                let sample = scatter
                    .as_ref()
                    .map_or([1.; 4], |image| sample_repeat(image, u / columns, v / rows));
                let mut coverage = mask
                    .as_ref()
                    .map_or(1., |image| sample_repeat(image, u * tile_u, v * tile_v)[0]);
                if scalar(data, "InvertMask", 0.) > 0. {
                    coverage = 1. - coverage;
                }
                coverage = lerp(
                    1.,
                    coverage,
                    scalar(data, "MaskIntensity", 1.).clamp(0., 1.),
                );
                let alpha = (if no_texture_alpha { 1. } else { sample[3] } * coverage)
                    .powf(exponent)
                    * max_alpha;
                image.put_pixel(
                    x,
                    y,
                    Rgba([
                        unit_to_byte(linear_to_srgb(lerp(dark[0], light[0], sample[0]))),
                        unit_to_byte(linear_to_srgb(lerp(dark[1], light[1], sample[0]))),
                        unit_to_byte(linear_to_srgb(lerp(dark[2], light[2], sample[0]))),
                        unit_to_byte(alpha),
                    ]),
                );
            }
        }
        fs::create_dir_all(output.parent().expect("cache output has a parent"))?;
        save_png(&image, &output)?;
        context.generated += 1;
    }
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(output.clone());
    material.alpha_mode = "BLEND";
    if scalar(data, "DoNotApplyLighting", 0.) > 0. {
        material.base_color_factor = [0., 0., 0., 1.];
        material.emissive_texture = Some(output);
        material.emissive_factor = [1.; 3];
    }
    material.preview_notes.push("Static blackbody FX samples the first authored atlas frame, smoke colors, mask tiling, mask inversion and alpha controls; runtime temperature color evolution, scrolling, depth softening and additive compositing are not simulated".into());
    Ok(material)
}

fn bake_pbr_layer(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
    enable_mask: bool,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = optional_texture(context.repository, data, "Diffuse")?;
    material.normal_texture = scaled_surface_normal(record, data, context)?;
    material.base_color_factor = gamma_color(data.get("TintColor"), [1.; 4]);
    material.base_color_factor[3] *= scalar(data, "LayerOpacity", 1.).clamp(0., 1.);
    let tile = scalar(data, "LayerTile", 1.);
    material.uv_transform = [
        tile,
        tile,
        scalar(data, "LayerOffsetU", 0.),
        scalar(data, "LayerOffsetV", 0.),
    ];
    if material.uv_transform.iter().any(|v| !v.is_finite()) {
        return Err(malformed("non-finite pbr_layer UV transform"));
    }
    if let Some(source) = optional_texture(context.repository, data, "RoughMetalBlend")? {
        let output = cache_directory(context.repository, record, context.size)?
            .join("pbr-layer-surface.png");
        let lock = bake_lock(&output);
        let _guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if output.is_file() {
            context.reused += 1;
        } else {
            let mut packed = load_rgba(&source, context.size)?;
            for pixel in Arc::make_mut(&mut packed).pixels_mut() {
                // Native packed surface: roughness R, metalness G, layer blend B.
                *pixel = Rgba([255, pixel[0], pixel[1], 255]);
            }
            fs::create_dir_all(output.parent().expect("cache output has a parent"))?;
            save_png(&packed, &output)?;
            context.generated += 1;
        }
        material.metallic_roughness_texture = Some(output);
        material.roughness_factor = 1.;
        material.metallic_factor = 1.;
    }
    if enable_mask {
        material.alpha_mode = "MASK";
        material.alpha_cutoff = Some(scalar(data, "AlphaThreshold", 0.5).clamp(0., 1.));
    } else if material.base_color_factor[3] < 1. {
        material.alpha_mode = "BLEND";
    }
    material.preview_notes.push("PBR layer uses authored diffuse, normal, packed roughness/metalness, tint, opacity and layer UV transform; multilayer microblend composition is not evaluated".into());
    Ok(material)
}

fn scaled_surface_normal(
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<Option<PathBuf>, PbrError> {
    let texture = optional_texture(context.repository, data, "Normal")?;
    let strength = scalar(data, "NormalStrength", 1.);
    if !strength.is_finite() {
        return Err(malformed("nonfinite native normal strength"));
    }
    if strength == 1. || texture.is_none() {
        return Ok(texture);
    }
    let input = load_rgba(
        texture.as_deref().expect("checked normal texture"),
        context.size,
    )?;
    Ok(Some(character::bake_map(
        record,
        "surface-normal-strength.png",
        context,
        |size| {
            rasterize(size, |x, y| {
                let mut n = decode_normal(sample_at(&input, x, y));
                n[0] *= strength;
                n[1] *= strength;
                encode_normal(normalize(n))
            })
        },
    )?))
}

fn bake_metal_base(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
    enable_mask: bool,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_texture = optional_texture(context.repository, data, "BaseColor")?;
    material.base_color_factor = gamma_color(data.get("BaseColorScale"), [1.0; 4]);
    material.normal_texture = scaled_surface_normal(record, data, context)?;
    let roughness_scale = scalar(data, "RoughnessScale", 1.0);
    let roughness_bias = scalar(data, "RoughnessBias", 0.0);
    let metallic_scale = scalar(data, "MetalnessScale", 1.0);
    let metallic_bias = scalar(data, "MetalnessBias", 0.0);
    let roughness = surface_texture(
        context.repository,
        data,
        "Roughness",
        roughness_scale,
        roughness_bias,
    )?;
    let metallic = surface_texture(
        context.repository,
        data,
        "Metalness",
        metallic_scale,
        metallic_bias,
    )?;
    if roughness.is_some() || metallic.is_some() {
        material.metallic_roughness_texture = Some(pack_metallic_roughness(
            record,
            roughness.as_deref(),
            metallic.as_deref(),
            roughness_scale,
            roughness_bias,
            metallic_scale,
            metallic_bias,
            context,
        )?);
        material.roughness_factor = 1.0;
        material.metallic_factor = 1.0;
    } else {
        material.roughness_factor = roughness_bias.clamp(0.0, 1.0);
        material.metallic_factor = metallic_bias.clamp(0.0, 1.0);
    }
    if enable_mask {
        material.alpha_mode = "MASK";
        let cutoff = scalar(data, "AlphaThreshold", 0.5).clamp(0.0, 1.0);
        material.alpha_cutoff = Some(cutoff);
    }
    if let Some(path) = optional_texture(context.repository, data, "Emissive")? {
        material.emissive_texture = Some(path);
        material.emissive_factor = color(
            data.get("EmissiveColor")
                .or_else(|| data.get("EmissiveColor1"))
                .or_else(|| data.get("DebugLightsIntensity")),
            [1.0, 1.0, 1.0, 1.0],
        )[..3]
            .try_into()
            .expect("three-element color slice");
        material.emissive_strength =
            scalar(data, "EmissiveEV", scalar(data, "Zone0EmissiveEV", 1.0)).max(0.0);
    }
    Ok(material)
}

#[expect(
    clippy::float_cmp,
    reason = "exact equality proves a saturated channel is constant"
)]
fn surface_texture(
    repository: &Path,
    data: &Map<String, Value>,
    key: &str,
    scale: f32,
    bias: f32,
) -> Result<Option<PathBuf>, PbrError> {
    // A fully saturated affine channel cannot depend on its texture. Old game
    // instances retain missing test maps even when the authored output is 1.
    if scale.is_finite() && bias.is_finite() && bias.clamp(0., 1.) == (scale + bias).clamp(0., 1.) {
        Ok(None)
    } else {
        optional_texture(repository, data, key)
    }
}

fn bake_decal(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.decal = Some(DecalShading {
        gradient_texture: None,
        gradient_intensity: 1.,
        gradient_uv: 1.,
        secondary_texture: None,
        secondary_color: [1.; 4],
        secondary_alpha: 0.,
        normal_alpha: scalar(data, "NormalAlpha", 1.).clamp(0., 1.),
        roughness_metalness_alpha: scalar(data, "RoughnessMetalnessAlpha", 1.).clamp(0., 1.),
        depth_threshold: scalar(data, "DepthThreshold", 1.).max(0.),
        normal_blending_mode: scalar(data, "NormalsBlendingMode", 1.),
        normal_alpha_texture: scalar(data, "UseNormalAlphaTex", 0.) > 0.,
    });
    material.base_color_texture = match optional_texture(context.repository, data, "DiffuseTexture")
    {
        Ok(texture) => texture,
        Err(PbrError::MissingTexture(_)) => {
            // A decal is an optional overlay, not the underlying body paint.
            // Keep missing pixels absent and explicitly report the omission.
            material.base_color_factor[3] = 0.0;
            material.alpha_mode = "BLEND";
            material.preview_notes.push(format!(
                "Omitted decal: missing authored DiffuseTexture {}",
                data.get("DiffuseTexture")
                    .and_then(resource_string)
                    .unwrap_or_default()
            ));
            return Ok(material);
        }
        Err(error) => return Err(error),
    };
    material.base_color_factor = gamma_color(data.get("DiffuseColor"), [1.0; 4]);
    material.normal_texture = match optional_texture(context.repository, data, "NormalTexture") {
        Ok(texture) => texture,
        Err(PbrError::MissingTexture(_)) => {
            material.preview_notes.push(format!(
                "Missing optional decal normal texture {}; neutral normal used",
                data.get("NormalTexture")
                    .and_then(resource_string)
                    .unwrap_or_default()
            ));
            None
        }
        Err(error) => return Err(error),
    };
    if scalar(data, "UseNormalAlphaTex", 0.) > 0. {
        let normal = material
            .normal_texture
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        let alpha = optional_texture(context.repository, data, "NormalAlphaTex")?
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        material.normal_texture = Some(character::bake_map(
            record,
            "decal-normal-alpha.png",
            context,
            |size| {
                RgbaImage::from_fn(size, size, |x, y| {
                    let mut pixel = normal
                        .as_ref()
                        .map_or(Rgba([128, 128, 255, 255]), |image| *image.get_pixel(x, y));
                    pixel[3] = alpha.as_ref().map_or(255, |image| image.get_pixel(x, y)[0]);
                    pixel
                })
            },
        )?);
    }
    let roughness = optional_texture(context.repository, data, "RoughnessTexture")?;
    let metallic = optional_texture(context.repository, data, "MetalnessTexture")?;
    if roughness.is_some() || metallic.is_some() {
        material.metallic_roughness_texture = Some(pack_metallic_roughness(
            record,
            roughness.as_deref(),
            metallic.as_deref(),
            scalar(data, "RoughnessScale", 1.0),
            scalar(data, "RoughnessBias", 0.0),
            scalar(data, "MetalnessScale", 1.0),
            scalar(data, "MetalnessBias", 0.0),
            context,
        )?);
        material.roughness_factor = 1.0;
        material.metallic_factor = 1.0;
    }
    material.base_color_factor[3] = scalar(data, "DiffuseAlpha", 1.0);
    if data
        .get("DiffuseTexture")
        .and_then(resource_string)
        .is_some_and(|path| path.contains("__basepenis_hairstyle_"))
    {
        // These recolorable atlases contain grayscale strands over a broad
        // low-alpha carrier. The carrier is not a dark skin-colour layer.
        let input = material
            .base_color_texture
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        material.base_color_texture = Some(character::bake_map(
            record,
            "strand-decal-coverage.png",
            context,
            |size| {
                rasterize(size, |x, y| {
                    let p = input.as_ref().map_or(Rgba([0; 4]), |m| *m.get_pixel(x, y));
                    let strand = f32::from(p[0].max(p[1]).max(p[2])) / 255.;
                    Rgba([255, 255, 255, unit_to_byte(strand * f32::from(p[3]) / 255.)])
                })
            },
        )?);
        material.preview_notes.push("Static recolored strand decal combines native grayscale strand intensity with alpha coverage; the low-alpha carrier is omitted from catalogue skin composition".into());
    }
    let influence = scalar(data, "SecondaryMaskInfluence", 0.).clamp(0., 1.);
    if influence > 0.
        && let Some(mask) = optional_texture(context.repository, data, "SecondaryMask")?
    {
        let mask = load_rgba(&mask, context.size)?;
        let diffuse = material
            .base_color_texture
            .as_deref()
            .map(|p| load_rgba(p, context.size))
            .transpose()?;
        let tile = scalar(data, "SecondaryMaskUVScale", 1.);
        material.base_color_texture = Some(character::bake_map(
            record,
            "decal-secondary-coverage.png",
            context,
            |size| {
                rasterize(size, |x, y| {
                    let mut pixel = diffuse
                        .as_ref()
                        .map_or(Rgba([255; 4]), |image| *image.get_pixel(x, y));
                    let coverage = sample_repeat(
                        &mask,
                        (x as f32 + 0.5) / size as f32 * tile,
                        (y as f32 + 0.5) / size as f32 * tile,
                    )[0];
                    // Native secondary masks suppress the decal where white;
                    // scars/bruises occupy their dark regions, not the atlas background.
                    pixel[3] = unit_to_byte(
                        f32::from(pixel[3]) / 255. * lerp(1., 1. - coverage, influence),
                    );
                    pixel
                })
            },
        )?);
    }
    material.alpha_mode = "BLEND";
    Ok(material)
}

fn bake_emissive_decal(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    material.base_color_factor = gamma_color(data.get("DiffuseColor2"), [0.0, 0.0, 0.0, 1.0]);
    material.emissive_texture = optional_texture(context.repository, data, "DiffuseTexture")?;
    material.emissive_factor = color(data.get("DiffuseColor"), [1.0, 1.0, 1.0, 1.0])[..3]
        .try_into()
        .expect("three-element color slice");
    material.emissive_strength = scalar(data, "EmissiveEV", 1.0).max(0.0);
    material.base_color_factor[3] = scalar(data, "DiffuseAlpha", 1.0).clamp(0.0, 1.0);
    material.alpha_mode = "BLEND";
    Ok(material)
}

fn bake_glass(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    let mask = optional_texture(context.repository, data, "MaskTexture")?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    let glass = optional_texture(context.repository, data, "GlassTint")?
        .map(|p| load_rgba(&p, context.size))
        .transpose()?;
    let tint = gamma_color(data.get("TintColor"), [0.75, 0.85, 0.95, 1.]);
    let paint = gamma_color(data.get("TintSurface"), [1.; 4]);
    let opacity = scalar(data, "Opacity", 1.).clamp(0., 1.);
    let mask_opacity = scalar(data, "MaskOpacity", 0.).clamp(0., 1.);
    material.base_color_texture = Some(character::bake_map(
        record,
        "glass-surface-coverage.png",
        context,
        |size| {
            rasterize(size, |x, y| {
                let overlay = mask
                    .as_ref()
                    .map_or([1., 1., 1., 0.], |m| sample_at(m, x, y));
                let amount = overlay[3] * mask_opacity;
                let base = glass.as_ref().map_or([1.; 4], |m| sample_at(m, x, y));
                let rgb: [f32; 3] = std::array::from_fn(|i| {
                    lerp(
                        srgb_to_linear(base[i]) * tint[i],
                        srgb_to_linear(overlay[i]) * paint[i],
                        amount,
                    )
                });
                Rgba([
                    unit_to_byte(linear_to_srgb(rgb[0])),
                    unit_to_byte(linear_to_srgb(rgb[1])),
                    unit_to_byte(linear_to_srgb(rgb[2])),
                    unit_to_byte(opacity * lerp(0.18, 1., amount)),
                ])
            })
        },
    )?);
    material.base_color_factor = [1.; 4];
    material.preview_notes.push("Static glass preserves separate tint and opaque label/surface-mask coverage; transmission is approximated by blended coverage against the studio background".into());
    material.normal_texture = scaled_surface_normal(record, data, context)?;
    let roughness = optional_texture(context.repository, data, "Roughness")?;
    if roughness.is_some() {
        material.metallic_roughness_texture = Some(pack_metallic_roughness(
            record,
            roughness.as_deref(),
            None,
            1.0,
            scalar(data, "GlassRoughnessBias", 0.0),
            0.0,
            0.0,
            context,
        )?);
        material.roughness_factor = 1.0;
    } else {
        material.roughness_factor = scalar(data, "GlassRoughnessBias", 0.08).clamp(0.0, 1.0);
    }
    material.alpha_mode = "BLEND";
    material.transmission = 1.0 - scalar(data, "MaskOpacity", 0.0).clamp(0.0, 1.0);
    Ok(material)
}

fn bake_screen(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    let texture = optional_texture(context.repository, data, "ParalaxTexture")?;
    material.base_color_texture.clone_from(&texture);
    material.emissive_texture = texture;
    material.base_color_factor = color(data.get("EmissiveColor"), [1.0; 4]);
    material.emissive_factor = material.base_color_factor[..3]
        .try_into()
        .expect("three-element color slice");
    material.emissive_strength = scalar(data, "EmissiveEV", scalar(data, "Emissive", 2.0)).max(0.0);
    material.metallic_factor = scalar(data, "Metalness", 0.0).clamp(0.0, 1.0);
    material.roughness_factor = scalar(data, "Roughness", 0.35).clamp(0.0, 1.0);
    Ok(material)
}

fn bake_window_parallax(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    let texture = optional_texture(context.repository, data, "RoomAtlas")?
        .or(optional_texture(context.repository, data, "LayerAtlas")?)
        .or(optional_texture(context.repository, data, "WindowTexture")?);
    material.base_color_texture.clone_from(&texture);
    material.emissive_texture = texture;
    material.base_color_factor = color(data.get("TintColorAtNight"), [1.0; 4]);
    material.emissive_factor = material.base_color_factor[..3]
        .try_into()
        .expect("three-element color slice");
    material.emissive_strength = scalar(data, "EmissiveEV", 1.0).max(0.0);
    material.normal_texture = optional_texture(context.repository, data, "Normal")?;
    material.roughness_factor = 0.35;
    Ok(material)
}

fn bake_invisible(name: String) -> PbrMaterial {
    let mut material = PbrMaterial::new(name);
    material.base_color_factor[3] = 0.0;
    material.alpha_mode = "BLEND";
    material
}

fn bake_signage(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    let texture = optional_texture(context.repository, data, "MainTexture")?;
    material.base_color_texture.clone_from(&texture);
    material.emissive_texture = texture;
    material.base_color_factor = color(data.get("ColorOneStart"), [1.0; 4]);
    material.emissive_factor = material.base_color_factor[..3]
        .try_into()
        .expect("three-element color slice");
    material.emissive_strength = (scalar(data, "EmissiveEV", 1.0) * 10.0).max(0.0);
    material.metallic_factor = scalar(data, "Metalness", 0.0).clamp(0.0, 1.0);
    material.roughness_factor = scalar(data, "Roughness", 0.3).clamp(0.0, 1.0);
    material.alpha_mode = "BLEND";
    Ok(material)
}

fn bake_hologram(
    name: String,
    data: &Map<String, Value>,
    context: &BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let mut material = PbrMaterial::new(name);
    let texture = optional_texture(context.repository, data, "Diffuse")?.or(optional_texture(
        context.repository,
        data,
        "Scanline",
    )?);
    material.base_color_texture.clone_from(&texture);
    material.emissive_texture = texture;
    material.base_color_factor = color(
        data.get("DotsColor").or_else(|| data.get("SurfaceColor")),
        [0.15, 0.65, 1.0, 1.0],
    );
    material.emissive_factor = material.base_color_factor[..3]
        .try_into()
        .expect("three-element color slice");
    material.emissive_strength = 30.0 * scalar(data, "GlowStrength", 1.0).max(0.1);
    material.base_color_factor[3] = scalar(data, "Opacity", 0.34).clamp(0.02, 1.0);
    material.roughness_factor = 1.0;
    material.alpha_mode = "BLEND";
    Ok(material)
}

#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "the bake loop keeps all per-pixel channels together; texture dimensions fit exactly enough in f32"
)]
fn bake_multilayered(
    name: String,
    record: &Value,
    data: &Map<String, Value>,
    context: &mut BakeContext<'_>,
) -> Result<PbrMaterial, PbrError> {
    let setup_depot = required_resource(data, "MultilayerSetup")?;
    let mask_depot = required_resource(data, "MultilayerMask")?;
    let mask_receipt = depot_to_path(context.repository, mask_depot, "masklist");
    if !mask_receipt.is_file() {
        return Err(PbrError::MissingTexture(mask_receipt));
    }
    let setup_path = depot_json_path(context.repository, setup_depot);
    let setup_document: Value = serde_json::from_slice(&fs::read(&setup_path)?)?;
    let setup = root_chunk(&setup_document)?;
    let layers = setup
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed(format!("{} has no layers", setup_path.display())))?;
    if layers.is_empty() {
        return Err(malformed(format!(
            "{} has no material layers",
            setup_path.display()
        )));
    }

    let cache = cache_directory(context.repository, record, context.size)?;
    let cache_lock = bake_lock(&cache);
    let _cache_guard = cache_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let base_path = cache.join("base.png");
    let mr_path = cache.join("metallic-roughness.png");
    let normal_path = cache.join("normal.png");
    if base_path.is_file() && mr_path.is_file() && normal_path.is_file() {
        context.reused += 3;
    } else {
        fs::create_dir_all(&cache)?;
        let masks = load_layer_masks(context.repository, mask_depot, layers.len(), context.size)?;
        let global_normal = optional_texture(context.repository, data, "GlobalNormal")?
            .map(|path| load_rgba(&path, context.size))
            .transpose()?;

        let mut surfaces = Vec::with_capacity(layers.len());
        for layer in layers {
            surfaces.push(load_multilayer_surface(
                context.repository,
                layer,
                context.size,
            )?);
        }

        let pixel_count = context
            .size
            .checked_mul(context.size)
            .ok_or_else(|| malformed("PBR texture dimensions overflow"))?;
        // The dedicated pool cannot steal mesh jobs or acquire bake locks.
        let pixels = parallel_pixels(pixel_count, |pixel_index| {
            let x = pixel_index % context.size;
            let y = pixel_index / context.size;
            let u = (x as f32 + 0.5) / context.size as f32;
            let v = (y as f32 + 0.5) / context.size as f32;
            let mut out_color = [0.0, 0.0, 0.0, 1.0];
            let mut out_roughness = 0.5;
            let mut out_metallic = 0.0;
            let mut out_normal = [0.0, 0.0, 1.0];
            for (index, surface) in surfaces.iter().enumerate() {
                let mask = if index == 0 {
                    1.0
                } else {
                    masks
                        .get(index)
                        .map_or(0.0, |image| f32::from(image.get_pixel(x, y)[0]) / 255.)
                };
                let opacity = surface.opacity;
                let microblend = surface.microblend.as_ref().map(|image| {
                    sample_repeat(
                        image,
                        u * surface.microblend_uv[0] + surface.microblend_uv[1],
                        v * surface.microblend_uv[0] + surface.microblend_uv[2],
                    )
                });
                let coverage = microblend.map_or(mask, |sample| {
                    microblend_coverage(mask, sample[3], surface.microblend_contrast)
                });
                let alpha = (coverage * opacity).clamp(0.0, 1.0);
                if alpha <= f32::EPSILON {
                    continue;
                }
                let sample_u = u * surface.uv[0] + surface.uv[1];
                let sample_v = v * surface.uv[0] + surface.uv[2];
                // These shared maps all have the bake dimensions. Resolve
                // tiled coordinates once for the four surface channels.
                let (tx, ty) = repeat_coordinates(context.size, context.size, sample_u, sample_v);
                let sampled_color = sample_at(&surface.color, tx, ty);
                let sampled_roughness = sample_at(&surface.roughness, tx, ty)[0];
                let sampled_metallic = sample_at(&surface.metallic, tx, ty)[0];
                let mut sampled_normal = decode_normal(sample_at(&surface.normal, tx, ty));
                sampled_normal[0] *= surface.normal_strength;
                sampled_normal[1] *= surface.normal_strength;
                if let Some(sample) = microblend {
                    // Microblend RG stores a signed normal perturbation; a
                    // negative strength reverses relief. Fully opaque layer
                    // coverage suppresses its boundary-normal contribution.
                    let strength = surface.microblend_normal_strength * f32::from(coverage < 1.);
                    sampled_normal[0] += (sample[0] * 2. - 1.) * strength;
                    sampled_normal[1] += (sample[1] * 2. - 1.) * strength;
                }
                sampled_normal = normalize(sampled_normal);
                for channel in 0..3 {
                    let source = SRGB_LINEAR[usize::from(unit_to_byte(sampled_color[channel]))]
                        * surface.color_scale[channel];
                    out_color[channel] = lerp(out_color[channel], source, alpha);
                    out_normal[channel] = lerp(out_normal[channel], sampled_normal[channel], alpha);
                }
                out_roughness = lerp(
                    out_roughness,
                    apply_levels(
                        apply_levels(sampled_roughness, surface.roughness_in),
                        surface.roughness_out,
                    ),
                    alpha,
                );
                out_metallic = lerp(
                    out_metallic,
                    apply_levels(
                        apply_levels(sampled_metallic, surface.metallic_in),
                        surface.metallic_out,
                    ),
                    alpha,
                );
            }
            if let Some(global) = &global_normal {
                let extra = decode_normal(sample_at(global, x, y));
                out_normal[0] += extra[0];
                out_normal[1] += extra[1];
            }
            out_normal = normalize(out_normal);
            (
                [
                    unit_to_byte(linear_to_srgb(out_color[0])),
                    unit_to_byte(linear_to_srgb(out_color[1])),
                    unit_to_byte(linear_to_srgb(out_color[2])),
                    255,
                ],
                [
                    255,
                    unit_to_byte(out_roughness),
                    unit_to_byte(out_metallic),
                    255,
                ],
                encode_normal(out_normal).0,
            )
        });
        let byte_capacity = usize::try_from(pixel_count)
            .map_err(|_| malformed("PBR texture exceeds platform size"))?
            .checked_mul(4)
            .ok_or_else(|| malformed("PBR texture byte size overflow"))?;
        let mut base_bytes = Vec::with_capacity(byte_capacity);
        let mut mr_bytes = Vec::with_capacity(byte_capacity);
        let mut normal_bytes = Vec::with_capacity(byte_capacity);
        for (base_pixel, mr_pixel, normal_pixel) in pixels {
            base_bytes.extend_from_slice(&base_pixel);
            mr_bytes.extend_from_slice(&mr_pixel);
            normal_bytes.extend_from_slice(&normal_pixel);
        }
        let base = RgbaImage::from_raw(context.size, context.size, base_bytes)
            .ok_or_else(|| malformed("could not construct baked base-color texture"))?;
        let mr = RgbaImage::from_raw(context.size, context.size, mr_bytes)
            .ok_or_else(|| malformed("could not construct baked metallic-roughness texture"))?;
        let normal = RgbaImage::from_raw(context.size, context.size, normal_bytes)
            .ok_or_else(|| malformed("could not construct baked normal texture"))?;
        save_png(&base, &base_path)?;
        save_png(&mr, &mr_path)?;
        save_png(&normal, &normal_path)?;
        context.generated += 3;
    }

    let mut material = PbrMaterial::new(name);
    material.base_color_texture = Some(base_path);
    material.metallic_roughness_texture = Some(mr_path);
    material.normal_texture = Some(normal_path);
    material.metallic_factor = 1.0;
    material.roughness_factor = 1.0;
    for fallback in setup
        .get("ResourceFallbacks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        material
            .preview_notes
            .push(format!("Native multilayer fallback: {fallback}"));
    }
    Ok(material)
}

struct MultilayerSurface {
    color: Arc<RgbaImage>,
    roughness: Arc<RgbaImage>,
    metallic: Arc<RgbaImage>,
    normal: Arc<RgbaImage>,
    microblend: Option<Arc<RgbaImage>>,
    normal_strength: f32,
    color_scale: [f32; 3],
    roughness_in: [f32; 2],
    roughness_out: [f32; 2],
    metallic_in: [f32; 2],
    metallic_out: [f32; 2],
    opacity: f32,
    uv: [f32; 3],
    microblend_uv: [f32; 3],
    microblend_contrast: f32,
    microblend_normal_strength: f32,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED material scalar values and image computations intentionally use f32"
)]
fn load_multilayer_surface(
    repository: &Path,
    layer: &Value,
    size: u32,
) -> Result<MultilayerSurface, PbrError> {
    let template_depot = resource_string(
        layer
            .get("material")
            .ok_or_else(|| malformed("multilayer layer has no material"))?,
    )
    .ok_or_else(|| malformed("multilayer layer material has no depot path"))?;
    let template_path = depot_json_path(repository, template_depot);
    let document: Value = serde_json::from_slice(&fs::read(&template_path)?)?;
    let template = root_chunk(&document)?;
    let overrides = template
        .get("overrides")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed(format!("{} has no overrides", template_path.display())))?;
    let defaults = template.get("defaultOverrides").unwrap_or(&Value::Null);

    let color_choice = red_name(layer.get("colorScale"))
        .or_else(|| red_name(defaults.get("colorScale")))
        .unwrap_or("null");
    let rough_in_choice = red_name(layer.get("roughLevelsIn"))
        .or_else(|| red_name(defaults.get("roughLevelsIn")))
        .unwrap_or("null");
    let rough_out_choice = red_name(layer.get("roughLevelsOut"))
        .or_else(|| red_name(defaults.get("roughLevelsOut")))
        .unwrap_or("null");
    let metal_in_choice = red_name(layer.get("metalLevelsIn"))
        .or_else(|| red_name(defaults.get("metalLevelsIn")))
        .unwrap_or("null");
    let metal_out_choice = red_name(layer.get("metalLevelsOut"))
        .or_else(|| red_name(defaults.get("metalLevelsOut")))
        .unwrap_or("null");
    let normal_choice = red_name(layer.get("normalStrength"))
        .or_else(|| red_name(defaults.get("normalStrength")))
        .unwrap_or("null");

    Ok(MultilayerSurface {
        microblend: optional_texture(
            repository,
            layer
                .as_object()
                .ok_or_else(|| malformed("multilayer layer object"))?,
            "microblend",
        )?
        .map(|path| load_rgba(&path, size))
        .transpose()?,
        color: load_rgba(
            &required_texture_from_value(repository, template, "colorTexture")?,
            size,
        )?,
        roughness: load_rgba(
            &required_texture_from_value(repository, template, "roughnessTexture")?,
            size,
        )?,
        metallic: load_rgba(
            &required_texture_from_value(repository, template, "metalnessTexture")?,
            size,
        )?,
        normal: load_rgba(
            &required_texture_from_value(repository, template, "normalTexture")?,
            size,
        )?,
        normal_strength: lookup_scalar_override(overrides, "normalStrength", normal_choice, 1.0),
        color_scale: lookup_override(overrides, "colorScale", color_choice, &[1.0, 1.0, 1.0])
            .try_into()
            .map_err(|_| malformed("invalid multilayer color override"))?,
        roughness_in: lookup_override(overrides, "roughLevelsIn", rough_in_choice, &[1.0, 0.0])
            .try_into()
            .map_err(|_| malformed("invalid multilayer roughness input override"))?,
        roughness_out: lookup_override(overrides, "roughLevelsOut", rough_out_choice, &[1.0, 0.0])
            .try_into()
            .map_err(|_| malformed("invalid multilayer roughness output override"))?,
        metallic_in: lookup_override(overrides, "metalLevelsIn", metal_in_choice, &[1.0, 0.0])
            .try_into()
            .map_err(|_| malformed("invalid multilayer metallic input override"))?,
        metallic_out: lookup_override(overrides, "metalLevelsOut", metal_out_choice, &[1.0, 0.0])
            .try_into()
            .map_err(|_| malformed("invalid multilayer metallic output override"))?,
        opacity: scalar_object(layer, "opacity", 1.),
        uv: [
            scalar_object(layer, "matTile", 1.)
                * template
                    .get("tilingMultiplier")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0) as f32,
            scalar_object(layer, "offsetU", 0.),
            scalar_object(layer, "offsetV", 0.),
        ],
        microblend_uv: [
            scalar_object(layer, "mbTile", scalar_object(layer, "matTile", 1.)),
            scalar_object(layer, "microblendOffsetU", 0.),
            scalar_object(layer, "microblendOffsetV", 0.),
        ],
        microblend_contrast: scalar_object(layer, "microblendContrast", 1.),
        microblend_normal_strength: scalar_object(layer, "microblendNormalStrength", 0.),
    })
}

fn load_layer_masks(
    repository: &Path,
    depot: &str,
    count: usize,
    size: u32,
) -> Result<Vec<Arc<RgbaImage>>, PbrError> {
    let source = depot_to_path(repository, depot, "png");
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| malformed("multilayer mask has no filename"))?;
    let directory = source.with_file_name(format!("{stem}_layers"));
    (0..count)
        .map(|index| {
            let path = directory.join(format!("{stem}_{index}.png"));
            if path.is_file() {
                load_rgba(&path, size)
            } else {
                Ok(Arc::new(ImageBuffer::from_pixel(
                    size,
                    size,
                    Rgba([0, 0, 0, 255]),
                )))
            }
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "the four scale/bias values are the explicit source material controls"
)]
fn pack_metallic_roughness(
    record: &Value,
    roughness: Option<&Path>,
    metallic: Option<&Path>,
    roughness_scale: f32,
    roughness_bias: f32,
    metallic_scale: f32,
    metallic_bias: f32,
    context: &mut BakeContext<'_>,
) -> Result<PathBuf, PbrError> {
    let output =
        cache_directory(context.repository, record, context.size)?.join("metallic-roughness.png");
    let output_lock = bake_lock(&output);
    let _output_guard = output_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if output.is_file() {
        context.reused += 1;
        return Ok(output);
    }
    let roughness_image = roughness
        .map(|path| load_rgba(path, context.size))
        .transpose()?;
    let metallic_image = metallic
        .map(|path| load_rgba(path, context.size))
        .transpose()?;
    let packed = rasterize(context.size, |x, y| {
        let rough = roughness_image
            .as_ref()
            .map_or(1.0, |image| f32::from(image.get_pixel(x, y)[0]) / 255.0);
        let metal = metallic_image
            .as_ref()
            .map_or(1.0, |image| f32::from(image.get_pixel(x, y)[0]) / 255.0);
        Rgba([
            255,
            unit_to_byte(rough * roughness_scale + roughness_bias),
            unit_to_byte(metal * metallic_scale + metallic_bias),
            255,
        ])
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    save_png(&packed, &output)?;
    context.generated += 1;
    Ok(output)
}

fn attach_materials(
    glb_path: &Path,
    appearance: &[String],
    materials: &[PbrMaterial],
) -> Result<(), PbrError> {
    let bytes = fs::read(glb_path)?;
    let (mut document, binary) = decode_glb(&bytes)?;
    let mut textures = Vec::new();
    let mut images = Vec::new();
    let glb_dir = glb_path.parent().unwrap_or_else(|| Path::new("."));
    let material_json = materials
        .iter()
        .map(|material| pbr_material_json(material, glb_dir, &mut textures, &mut images))
        .collect::<Result<Vec<_>, PbrError>>()?;
    let root = document
        .as_object_mut()
        .ok_or_else(|| malformed("GLB root is not an object"))?;
    root.insert("materials".to_owned(), Value::Array(material_json));
    if !textures.is_empty() {
        root.insert("textures".to_owned(), Value::Array(textures));
        root.insert("images".to_owned(), Value::Array(images));
        root.insert(
            "samplers".to_owned(),
            json!([{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]),
        );
    }
    let mut extensions = Vec::new();
    if materials.iter().any(|material| material.transmission > 0.0) {
        extensions.push("KHR_materials_transmission");
    }
    if materials.iter().any(|material| {
        material.emissive_strength > 1.0 && material.emissive_factor.iter().any(|v| *v > 0.)
    }) {
        extensions.push("KHR_materials_emissive_strength");
    }
    if materials.iter().any(PbrMaterial::has_uv_transform) {
        extensions.push("KHR_texture_transform");
    }
    if !extensions.is_empty() {
        root.insert("extensionsUsed".to_owned(), json!(extensions));
    }
    let material_indices: HashMap<_, _> = materials
        .iter()
        .enumerate()
        .map(|(index, material)| (material.name.as_str(), index))
        .collect();
    let meshes = root
        .get_mut("meshes")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| malformed("GLB has no meshes"))?;
    for (index, mesh) in meshes.iter_mut().enumerate() {
        // LOD filtering removes meshes but retains their original chunk IDs.
        // Appearance arrays describe those IDs, not the filtered mesh count.
        let chunk_index = mesh
            .get("name")
            .and_then(Value::as_str)
            .and_then(|name| name.strip_prefix("submesh_"))
            .and_then(|name| name.split('_').next())
            .and_then(|index| index.parse::<usize>().ok())
            .unwrap_or(index);
        let name = material_for_chunk(appearance, chunk_index);
        let material_index = material_indices
            .get(name)
            .ok_or_else(|| malformed(format!("no baked material for {name:?}")))?;
        for primitive in mesh
            .get_mut("primitives")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| malformed("GLB mesh has no primitives"))?
        {
            let primitive = primitive
                .as_object_mut()
                .ok_or_else(|| malformed("GLB primitive is not an object"))?;
            primitive.insert("material".to_owned(), json!(material_index));
            // Cyberpunk uses its vertex-color stream as shader data rather than
            // the glTF base-color multiplier. Keeping COLOR_0 would desaturate
            // baked textures in standards-compliant importers.
            if let Some(attributes) = primitive
                .get_mut("attributes")
                .and_then(Value::as_object_mut)
            {
                attributes.remove("COLOR_0");
            }
        }
    }
    fs::write(glb_path, encode_glb(&document, binary.as_deref())?)?;
    Ok(())
}

fn pbr_material_json(
    material: &PbrMaterial,
    glb_dir: &Path,
    textures: &mut Vec<Value>,
    images: &mut Vec<Value>,
) -> Result<Value, PbrError> {
    let mut pbr = Map::new();
    pbr.insert(
        "baseColorFactor".to_owned(),
        json!(material.base_color_factor),
    );
    pbr.insert("metallicFactor".to_owned(), json!(material.metallic_factor));
    pbr.insert(
        "roughnessFactor".to_owned(),
        json!(material.roughness_factor),
    );
    if let Some(path) = &material.base_color_texture {
        pbr.insert(
            "baseColorTexture".to_owned(),
            texture_reference(path, glb_dir, textures, images)?,
        );
    }
    if let Some(path) = &material.metallic_roughness_texture {
        pbr.insert(
            "metallicRoughnessTexture".to_owned(),
            texture_reference(path, glb_dir, textures, images)?,
        );
    }
    let mut value = json!({
        "name": material.name,
        "doubleSided": material.double_sided,
        "alphaMode": material.alpha_mode,
        "pbrMetallicRoughness": pbr,
    });
    if !material.preview_notes.is_empty() {
        value["extras"]["ghostline_preview_notes"] = json!(material.preview_notes);
    }
    if let Some(cutoff) = material.alpha_cutoff {
        value["alphaCutoff"] = json!(cutoff);
    }
    if let Some(path) = &material.normal_texture {
        value["normalTexture"] = texture_reference(path, glb_dir, textures, images)?;
    }
    if let Some(path) = &material.emissive_texture {
        value["emissiveTexture"] = texture_reference(path, glb_dir, textures, images)?;
    }
    if material.emissive_factor.iter().any(|v| *v > 0.) {
        value["emissiveFactor"] = json!(material.emissive_factor);
        if material.emissive_strength > 1.0 {
            value["extensions"]["KHR_materials_emissive_strength"] =
                json!({"emissiveStrength": material.emissive_strength});
        }
    }
    if material.transmission > 0.0 {
        value["extensions"]["KHR_materials_transmission"] =
            json!({"transmissionFactor": material.transmission});
    }
    if material.has_uv_transform() {
        for pointer in [
            "/pbrMetallicRoughness/baseColorTexture",
            "/pbrMetallicRoughness/metallicRoughnessTexture",
            "/normalTexture",
            "/emissiveTexture",
        ] {
            if let Some(texture) = value.pointer_mut(pointer) {
                texture["extensions"]["KHR_texture_transform"] = json!({
                    "scale": &material.uv_transform[..2],
                    "offset": &material.uv_transform[2..],
                });
            }
        }
    }
    Ok(value)
}

fn texture_reference(
    path: &Path,
    glb_dir: &Path,
    textures: &mut Vec<Value>,
    images: &mut Vec<Value>,
) -> Result<Value, PbrError> {
    let relative = pathdiff::diff_paths(path, glb_dir).ok_or_else(|| {
        malformed(format!(
            "cannot make {} relative to {}",
            path.display(),
            glb_dir.display()
        ))
    })?;
    let uri = relative.to_string_lossy().replace('\\', "/");
    let image_index = images.len();
    images.push(json!({"uri": uri}));
    let texture_index = textures.len();
    textures.push(json!({"sampler": 0, "source": image_index}));
    Ok(json!({"index": texture_index}))
}

fn optional_texture(
    repository: &Path,
    data: &Map<String, Value>,
    key: &str,
) -> Result<Option<PathBuf>, PbrError> {
    let Some(depot) = data.get(key).and_then(resource_string) else {
        return Ok(None);
    };
    if depot.is_empty() {
        return Ok(None);
    }
    let path = depot_to_path(repository, depot, "png");
    if !path.is_file() {
        return Err(PbrError::MissingTexture(path));
    }
    Ok(Some(path))
}

fn required_texture_from_value(
    repository: &Path,
    value: &Value,
    key: &str,
) -> Result<PathBuf, PbrError> {
    let depot = value
        .get(key)
        .and_then(resource_string)
        .ok_or_else(|| malformed(format!("missing texture {key}")))?;
    let path = depot_to_path(repository, depot, "png");
    if !path.is_file() {
        return Err(PbrError::MissingTexture(path));
    }
    Ok(path)
}

fn required_resource<'a>(data: &'a Map<String, Value>, key: &str) -> Result<&'a str, PbrError> {
    data.get(key)
        .and_then(resource_string)
        .ok_or_else(|| malformed(format!("missing resource {key}")))
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, PbrError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| malformed(format!("missing string {key}")))
}

fn resource_string(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("$value").and_then(Value::as_str))
        .or_else(|| value.get("DepotPath").and_then(resource_string))
        .filter(|path| !path.is_empty() && *path != "0")
}

fn red_name(value: Option<&Value>) -> Option<&str> {
    value.and_then(resource_string)
}

fn depot_to_path(repository: &Path, depot: &str, extension: &str) -> PathBuf {
    let mut path = repository.to_path_buf();
    for component in Path::new(&depot.replace('\\', "/")).components() {
        if let Component::Normal(part) = component {
            path.push(part);
        }
    }
    path.set_extension(extension);
    path
}

fn depot_json_path(repository: &Path, depot: &str) -> PathBuf {
    let path = depot_to_path(
        repository,
        depot,
        Path::new(depot)
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or(""),
    );
    PathBuf::from(format!("{}.json", path.display()))
}

fn root_chunk(document: &Value) -> Result<&Value, PbrError> {
    document
        .pointer("/Data/RootChunk")
        .ok_or_else(|| malformed("resource JSON has no Data.RootChunk"))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED material scalar values are represented as f32 by the renderer"
)]
fn scalar(data: &Map<String, Value>, key: &str, default: f32) -> f32 {
    data.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |value| value as f32)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED material scalar values are represented as f32 by the renderer"
)]
fn scalar_object(data: &Value, key: &str, default: f32) -> f32 {
    data.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |value| value as f32)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED color components are represented as f32 by the renderer"
)]
fn color(value: Option<&Value>, default: [f32; 4]) -> [f32; 4] {
    let Some(value) = value else {
        return default;
    };
    let byte_color = ["Red", "Green", "Blue", "Alpha"]
        .iter()
        .any(|key| value.get(*key).is_some());
    let names = if byte_color {
        ["Red", "Green", "Blue", "Alpha"]
    } else {
        ["X", "Y", "Z", "W"]
    };
    let mut result = default;
    for (index, name) in names.iter().enumerate() {
        if let Some(component) = value.get(name).and_then(Value::as_f64) {
            result[index] = component as f32 / if byte_color { 255.0 } else { 1.0 };
        }
    }
    result
}

fn gamma_color(value: Option<&Value>, default: [f32; 4]) -> [f32; 4] {
    let mut result = color(value, default);
    // RED Color is byte/sRGB; Vector4 tint parameters are already linear.
    // Applying gamma again turns native dark plastics almost completely black.
    if value.is_some_and(|v| {
        v.get("Red").is_some() || v.get("Green").is_some() || v.get("Blue").is_some()
    }) {
        for component in &mut result[..3] {
            *component = component.max(0.0).powf(2.2);
        }
    }
    result
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED material override values are represented as f32 by the renderer"
)]
fn lookup_override(
    overrides: &Map<String, Value>,
    kind: &str,
    selected: &str,
    default: &[f32],
) -> Vec<f32> {
    overrides
        .get(kind)
        .and_then(Value::as_array)
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| red_name(entry.get("n")) == Some(selected))
        })
        .and_then(|entry| {
            entry
                .pointer("/v/Elements")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_f64)
                        .map(|value| value as f32)
                        .collect()
                })
        })
        .filter(|values: &Vec<f32>| values.len() == default.len())
        .unwrap_or_else(|| default.to_vec())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "RED normal strength is a 32-bit scalar"
)]
fn lookup_scalar_override(
    overrides: &Map<String, Value>,
    kind: &str,
    selected: &str,
    default: f32,
) -> f32 {
    overrides
        .get(kind)
        .and_then(Value::as_array)
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| red_name(entry.get("n")) == Some(selected))
        })
        // Omitted v is the scalar class constructor's zero, not a missing override.
        .map_or(default, |entry| {
            entry.get("v").and_then(Value::as_f64).unwrap_or(0.0) as f32
        })
}

fn cache_directory(repository: &Path, record: &Value, size: u32) -> Result<PathBuf, PbrError> {
    let mut hasher = Sha1::new();
    hasher.update(b"ghostline-pbr-v9-microblend");
    hasher.update(serde_json::to_vec(record)?);
    hasher.update(size.to_le_bytes());
    Ok(repository
        .join("_ghostline_pbr")
        .join(format!("{:x}", hasher.finalize())))
}

fn bake_lock(path: &Path) -> Arc<Mutex<()>> {
    let mut locks = BAKE_LOCKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Arc::clone(
        locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

fn load_rgba(path: &Path, size: u32) -> Result<Arc<RgbaImage>, PbrError> {
    const BUDGET: usize = 128 * 1024 * 1024;
    let metadata = fs::metadata(path).map_err(|_| PbrError::MissingTexture(path.to_path_buf()))?;
    let key = (
        path.to_path_buf(),
        size,
        metadata.len(),
        metadata.modified()?,
    );
    let lock = bake_lock(&path.with_extension(format!("scaled-{size}")));
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(image) = SCALED_TEXTURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entries
        .get(&key)
        .cloned()
    {
        return Ok(image);
    }
    let image = image::open(path)?.into_rgba8();
    let image = if image.width() == size && image.height() == size {
        image
    } else {
        resize_rgba(image, size)?
    };
    let image = Arc::new(image);
    let bytes = image.as_raw().len();
    if bytes <= BUDGET / 4 {
        let mut cache = SCALED_TEXTURES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while cache.bytes + bytes > BUDGET {
            let Some(old) = cache.entries.keys().next().cloned() else {
                break;
            };
            if let Some(removed) = cache.entries.remove(&old) {
                cache.bytes -= removed.as_raw().len();
            }
        }
        cache.bytes += bytes;
        cache.entries.insert(key, Arc::clone(&image));
    }
    Ok(image)
}

// Independent channel filtering matches the existing material-map semantics:
// alpha is coverage, while RGB can contain normals or packed scalar channels.
fn resize_rgba(image: RgbaImage, size: u32) -> Result<RgbaImage, PbrError> {
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image,
    };
    let source = Image::from_vec_u8(
        image.width(),
        image.height(),
        image.into_raw(),
        PixelType::U8x4,
    )
    .map_err(|error| malformed(format!("resize source: {error}")))?;
    let mut target = Image::new(size, size, PixelType::U8x4);
    Resizer::new()
        .resize(
            &source,
            &mut target,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))
                .use_alpha(false),
        )
        .map_err(|error| malformed(format!("resize material texture: {error}")))?;
    RgbaImage::from_raw(size, size, target.into_vec())
        .ok_or_else(|| malformed("resized material texture has invalid dimensions"))
}

static SRGB_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        srgb_to_linear(f32::from(u8::try_from(index).expect("byte lookup")) / 255.)
    })
});

fn save_png(image: &RgbaImage, path: &Path) -> Result<(), PbrError> {
    image::codecs::png::PngEncoder::new_with_quality(
        fs::File::create(path)?,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Adaptive,
    )
    .write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(())
}

fn sample_repeat(image: &RgbaImage, u: f32, v: f32) -> [f32; 4] {
    let (x, y) = repeat_coordinates(image.width(), image.height(), u, v);
    sample_at(image, x, y)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "wrapped texture coordinates are bounded by image dimensions"
)]
fn repeat_coordinates(width: u32, height: u32, u: f32, v: f32) -> (u32, u32) {
    let x = (u.rem_euclid(1.0) * width as f32).floor() as u32 % width;
    let y = (v.rem_euclid(1.0) * height as f32).floor() as u32 % height;
    (x, y)
}

fn sample_at(image: &RgbaImage, x: u32, y: u32) -> [f32; 4] {
    let pixel = image.get_pixel(x, y);
    [
        f32::from(pixel[0]) / 255.0,
        f32::from(pixel[1]) / 255.0,
        f32::from(pixel[2]) / 255.0,
        f32::from(pixel[3]) / 255.0,
    ]
}

fn decode_normal(pixel: [f32; 4]) -> [f32; 3] {
    let x = pixel[0] * 2.0 - 1.0;
    let y = pixel[1] * 2.0 - 1.0;
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    normalize([x, y, z])
}

fn encode_normal(value: [f32; 3]) -> Rgba<u8> {
    Rgba([
        unit_to_byte(value[0] * 0.5 + 0.5),
        unit_to_byte(value[1] * 0.5 + 0.5),
        unit_to_byte(value[2] * 0.5 + 0.5),
        255,
    ])
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to the exact u8 range before conversion"
)]
fn unit_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn lerp(left: f32, right: f32, factor: f32) -> f32 {
    left + (right - left) * factor
}

fn apply_levels(value: f32, levels: [f32; 2]) -> f32 {
    (value * levels[0] + levels[1]).clamp(0.0, 1.0)
}

/// Combines authored layer coverage with the microblend alpha boundary.
/// Contrast sets the transition width; opacity is applied by the layer caller.
fn microblend_coverage(mask: f32, alpha: f32, contrast: f32) -> f32 {
    let width = contrast.max(0.00001);
    let mixed = lerp(alpha + mask - 1., mask, width.clamp(0., 1.));
    (mixed / width).clamp(0., 1.)
}

#[cfg(test)]
mod microblend_tests {
    use super::microblend_coverage;

    #[test]
    fn microblend_adds_detail_to_partial_masks_without_leaking_outside_layers() {
        for contrast in [0., 0.35, 1.] {
            assert_eq!(microblend_coverage(0., 1., contrast), 0.);
            assert_eq!(microblend_coverage(1., 0., contrast), 1.);
        }
        assert_eq!(microblend_coverage(0.5, 0., 0.35), 0.);
        assert_eq!(microblend_coverage(0.5, 1., 0.35), 1.);
    }
}

fn material_for_chunk(materials: &[String], index: usize) -> &str {
    if materials.is_empty() {
        return "";
    }
    &materials[index % materials.len()]
}

#[cfg(test)]
fn selected_material_names(
    geometry: &Value,
    materials: &[String],
) -> Result<HashSet<String>, PbrError> {
    if materials.is_empty() {
        return Err(malformed("appearance has no chunk materials"));
    }
    let meshes = geometry
        .get("meshes")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("GLB has no meshes"))?;
    Ok(meshes
        .iter()
        .enumerate()
        .map(|(index, mesh)| {
            let chunk = mesh
                .get("name")
                .and_then(Value::as_str)
                .and_then(|n| n.strip_prefix("submesh_"))
                .and_then(|n| n.split('_').next())
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(index);
            material_for_chunk(materials, chunk).to_owned()
        })
        .collect())
}

fn decode_glb(bytes: &[u8]) -> Result<(Value, Option<Vec<u8>>), PbrError> {
    if bytes.len() < 20 || u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != GLB_MAGIC {
        return Err(malformed("input is not a GLB"));
    }
    let mut offset = 12;
    let mut document = None;
    let mut binary = None;
    while offset + 8 <= bytes.len() {
        let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        offset += 8;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| malformed("GLB chunk length overflow"))?;
        let payload = bytes
            .get(offset..end)
            .ok_or_else(|| malformed("GLB chunk exceeds file"))?;
        match kind {
            GLB_JSON_CHUNK => document = Some(serde_json::from_slice(payload)?),
            GLB_BIN_CHUNK => binary = Some(payload.to_vec()),
            _ => {}
        }
        offset = end;
    }
    Ok((
        document.ok_or_else(|| malformed("GLB has no JSON chunk"))?,
        binary,
    ))
}

fn encode_glb(document: &Value, binary: Option<&[u8]>) -> Result<Vec<u8>, PbrError> {
    let mut json_bytes = serde_json::to_vec(document)?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let binary_length = binary.map_or(0, |value| (value.len() + 3) & !3);
    let total = 12
        + 8
        + json_bytes.len()
        + if binary.is_some() {
            8 + binary_length
        } else {
            0
        };
    let mut output = Vec::with_capacity(total);
    output.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(total)
            .map_err(|_| malformed("GLB exceeds four GiB"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u32::try_from(json_bytes.len())
            .map_err(|_| malformed("GLB JSON exceeds four GiB"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(&GLB_JSON_CHUNK.to_le_bytes());
    output.extend_from_slice(&json_bytes);
    if let Some(binary) = binary {
        output.extend_from_slice(
            &u32::try_from(binary_length)
                .map_err(|_| malformed("GLB binary exceeds four GiB"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&GLB_BIN_CHUNK.to_le_bytes());
        output.extend_from_slice(binary);
        output.resize(total, 0);
    }
    Ok(output)
}

fn malformed(reason: impl Into<String>) -> PbrError {
    PbrError::Malformed(reason.into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn dedicated_pixel_pool_preserves_pixels_under_outer_cache_lock() {
        use rayon::prelude::*;
        let pixel = |x: u32, y: u32| {
            image::Rgba([
                u8::try_from(x % 256).unwrap(),
                u8::try_from(y % 256).unwrap(),
                u8::try_from((x ^ y) % 256).unwrap(),
                255,
            ])
        };
        let expected = image::RgbaImage::from_fn(512, 512, pixel);
        let lock = std::sync::Mutex::new(());
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        outer.install(|| {
            (0..4).into_par_iter().for_each(|_| {
                let _guard = lock.lock().unwrap();
                assert_eq!(super::rasterize(512, pixel), expected);
            })
        });
    }
    use super::{BakeContext, apply_levels, bake_material, material_for_chunk};

    #[test]
    fn typed_appearance_bake_selects_original_chunks_without_geometry_export() {
        let folder = tempfile::tempdir().unwrap();
        let sidecar = folder.path().join("mesh.Material.json");
        let document = serde_json::json!({
            "MaterialRepo": folder.path(),
            "Appearances": {"gold": ["unused", "body"]},
            "Materials": [
                {"Name": "unused", "MaterialTemplate": "unsupported.mt", "Data": {}},
                {"Name": "body", "MaterialTemplate": "engine\\materials\\metal_base.remt", "Data": {}}
            ]
        });
        std::fs::write(&sidecar, serde_json::to_vec(&document).unwrap()).unwrap();
        let baked = super::bake_appearance(&sidecar, "gold", &[1, 3], Some(4)).unwrap();
        assert_eq!(baked.materials.len(), 1);
        assert_eq!(baked.materials[0].name, "body");
        assert_eq!(baked.chunk_materials, ["unused", "body"]);
        assert!(super::bake_appearance(&sidecar, "missing", &[1], Some(4)).is_err());
        assert!(super::bake_appearance(&sidecar, "gold", &[], Some(4)).is_err());
    }

    #[test]
    fn filtered_lod_geometry_does_not_bake_unused_materials() {
        let geometry = serde_json::json!({"meshes":[{"name":"submesh_0"},{"name":"submesh_5"}]});
        let names = vec![
            "base".to_owned(),
            "unused_missing".to_owned(),
            "chrome".to_owned(),
        ];
        let selected = super::selected_material_names(&geometry, &names).unwrap();
        assert!(selected.contains("base") && selected.contains("chrome"));
        assert!(!selected.contains("unused_missing"));
    }

    #[test]
    fn missing_decal_is_explicitly_omitted_but_missing_body_paint_still_fails() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 1,
            generated: 0,
            reused: 0,
        };
        let decal = serde_json::json!({"Name":"decal","MaterialTemplate":r"base\materials\vehicle_mesh_decal.mt","Data":{"DiffuseTexture":r"missing\decal.xbm"}});
        let material = bake_material(&decal, &mut context).unwrap();
        assert!(material.base_color_factor[3].abs() < f32::EPSILON);
        assert!(material.preview_notes[0].contains("Omitted decal"));
        let body = serde_json::json!({"Name":"body","MaterialTemplate":r"engine\materials\metal_base.remt","Data":{"BaseColor":r"missing\body.xbm"}});
        assert!(bake_material(&body, &mut context).is_err());
    }

    #[test]
    fn inactive_and_normal_only_decals_do_not_turn_into_white_paint() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 1,
            generated: 0,
            reused: 0,
        };
        let mut record = serde_json::json!({"Name":"off","MaterialTemplate":r"base\materials\vehicle_mesh_decal.mt","Data":{"DiffuseAlpha":0.0,"NormalAlpha":0.0,"DiffuseTexture":r"missing\unused.xbm"}});
        let inactive = bake_material(&record, &mut context).unwrap();
        assert!(inactive.base_color_factor[3].abs() < f32::EPSILON);
        assert!(
            inactive
                .preview_notes
                .iter()
                .any(|note| note.contains("Omitted decal"))
        );
        record["Data"]["NormalAlpha"] = serde_json::json!(1.0);
        let normal_only = bake_material(&record, &mut context).unwrap();
        assert!(normal_only.base_color_factor[3].abs() < f32::EPSILON);
        assert_eq!(normal_only.decal.unwrap().normal_alpha, 1.0);
    }

    #[test]
    fn runtime_ui_without_bound_pixels_is_transparent_but_missing_authored_texture_is_an_error() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 1,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"ui","MaterialTemplate":r"base\fx\shaders\parallaxscreen_transparent_ui.mt","Data":{}});
        let material = bake_material(&record, &mut context).unwrap();
        assert!(material.base_color_factor[3].abs() < f32::EPSILON);
        let mut record = record;
        record["Data"]["ParalaxTexture"] = serde_json::json!(r"missing\authored.xbm");
        assert!(bake_material(&record, &mut context).is_err());
    }

    #[test]
    fn omitted_normal_strength_override_is_zero() {
        let value = serde_json::json!({"normalStrength":[{"n":{"$value":"smooth"}},{"n":{"$value":"grain"},"v":0.15}]});
        let overrides = value.as_object().unwrap();
        assert!(
            super::lookup_scalar_override(overrides, "normalStrength", "smooth", 1.0).abs()
                < f32::EPSILON
        );
        assert!(
            (super::lookup_scalar_override(overrides, "normalStrength", "grain", 1.0) - 0.15).abs()
                < 1e-6
        );
        assert!(
            (super::lookup_scalar_override(overrides, "normalStrength", "missing", 1.0) - 1.0)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn converts_byte_tints_without_scaling_float_vectors() {
        let bytes = super::color(
            Some(&serde_json::json!({"Red":255,"Green":128,"Blue":0,"Alpha":255})),
            [0.0; 4],
        );
        assert!((bytes[0] - 1.0).abs() < f32::EPSILON);
        assert!((bytes[1] - 128.0 / 255.0).abs() < 1e-6);
        let floats = super::color(
            Some(&serde_json::json!({"X":0.5,"Y":1.0,"Z":0.0,"W":1.0})),
            [0.0; 4],
        );
        assert!(
            floats
                .iter()
                .zip([0.5, 1.0, 0.0, 1.0])
                .all(|(actual, expected)| (actual - expected).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn static_vehicle_glass_preserves_transparent_coverage() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 4,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"window","MaterialTemplate":r"base\materials\vehicle_glass.mt","Data":{"Opacity":1.0}});
        let material = bake_material(&record, &mut context).unwrap();
        assert_eq!(material.alpha_mode, "BLEND");
        assert!((material.base_color_factor[3] - 0.18).abs() < 1e-6);
        assert!(material.roughness_factor > 0.0);
    }

    #[test]
    fn runtime_heat_is_unbound_in_a_parked_vehicle() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 4,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"overheat","MaterialTemplate":r"base\fx\_shaders\mesh_decal__blackbody.mt","Data":{"Temperature":25.0,"UseFloatParam":1.0,"EmissiveEV":5.0}});
        let material = bake_material(&record, &mut context).unwrap();
        assert_eq!(material.base_color_factor[3], 0.);
        assert_eq!(material.alpha_mode, "BLEND");
        assert!(!material.preview_notes.is_empty());
    }

    #[test]
    fn pbr_layer_preserves_surface_channels_and_uv_transform() {
        let folder = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([64, 192, 17, 255]))
            .save(folder.path().join("surface.png"))
            .unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 2,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"tape","MaterialTemplate":r"base\materials\pbr_layer.remt","Data":{
            "RoughMetalBlend":"surface.xbm", "LayerTile":10., "LayerOffsetU":0.25, "LayerOffsetV":-0.5,
            "LayerOpacity":0.75
        }});
        let material = bake_material(&record, &mut context).unwrap();
        assert_eq!(material.uv_transform, [10., 10., 0.25, -0.5]);
        assert_eq!(material.alpha_mode, "BLEND");
        let surface = image::open(material.metallic_roughness_texture.as_ref().unwrap())
            .unwrap()
            .into_rgba8();
        assert_eq!(surface.get_pixel(0, 0).0, [255, 64, 192, 255]);
        let mut textures = Vec::new();
        let mut images = Vec::new();
        let exported =
            super::pbr_material_json(&material, folder.path(), &mut textures, &mut images).unwrap();
        assert_eq!(
            exported["pbrMetallicRoughness"]["metallicRoughnessTexture"]["extensions"]["KHR_texture_transform"]
                ["scale"],
            serde_json::json!([10., 10.])
        );
    }

    #[test]
    fn static_thruster_keeps_authored_mask_opacity_and_unlit_color() {
        let folder = tempfile::tempdir().unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 255, 255, 0]))
            .save(folder.path().join("scatter.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([128, 128, 128, 255]))
            .save(folder.path().join("mask.png"))
            .unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 2,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"thruster","MaterialTemplate":r"base\fx\shaders\blackbodyradiation.mt","Data":{
            "FireScatterAlpha":"scatter.xbm", "MaskTexture":"mask.xbm", "NoAlphaOnTexture":1.,
            "AlphaExponent":2., "maxAlpha":0.56, "DoNotApplyLighting":1.,
            "LightSmoke":{"Red":255,"Green":171,"Blue":122,"Alpha":255}
        }});
        let material = bake_material(&record, &mut context).unwrap();
        assert_eq!(material.alpha_mode, "BLEND");
        assert_eq!(material.base_color_factor, [0., 0., 0., 1.]);
        assert_eq!(material.emissive_factor, [1.; 3]);
        let image = image::open(material.base_color_texture.unwrap())
            .unwrap()
            .into_rgba8();
        let pixel = image.get_pixel(0, 0).0;
        for (actual, expected) in pixel[..3].iter().zip([255_u8, 171, 122]) {
            assert!(actual.abs_diff(expected) <= 1);
        }
        assert!((35..=37).contains(&pixel[3]));
    }

    #[test]
    fn neon_constant_emission_exports_without_a_texture() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 2,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"tube","MaterialTemplate":r"base\fx\shaders\neon_tubes.mt","Data":{
            "color":{"Red":255,"Green":255,"Blue":255,"Alpha":255}, "EmissiveEV":8.,
            "TextureIntensity":0.,"tex1":"missing-unused-texture.xbm"
        }});
        let material = bake_material(&record, &mut context).unwrap();
        assert_eq!(material.emissive_factor, [1.; 3]);
        assert!((material.emissive_strength - 256.).abs() < f32::EPSILON);
        assert!(material.emissive_texture.is_none());
        let exported =
            super::pbr_material_json(&material, folder.path(), &mut Vec::new(), &mut Vec::new())
                .unwrap();
        assert_eq!(exported["emissiveFactor"], serde_json::json!([1., 1., 1.]));
        assert_eq!(
            exported["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
            256.
        );
    }

    #[test]
    fn vehicle_lights_do_not_invent_runtime_emission() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 4,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"light","MaterialTemplate":r"base\materials\vehicle_lights.mt","EnableMask":true,"Data":{"DebugLightsIntensity":{"X":1.0,"Y":1.0,"Z":1.0,"W":1.0}}});
        let material = bake_material(&record, &mut context).unwrap();
        assert!(
            material
                .emissive_factor
                .iter()
                .all(|value| value.abs() < f32::EPSILON)
        );
        assert_eq!(material.alpha_mode, "MASK");
    }

    #[test]
    fn vehicle_paint_requires_its_multilayer_setup() {
        let folder = tempfile::tempdir().unwrap();
        let mut context = BakeContext {
            repository: folder.path(),
            size: 4,
            generated: 0,
            reused: 0,
        };
        let record = serde_json::json!({"Name":"paint","MaterialTemplate":r"base\materials\vehicle_destr_blendshape.mt","Data":{}});
        let error = bake_material(&record, &mut context).unwrap_err();
        assert!(error.to_string().contains("MultilayerSetup"));
    }

    #[test]
    fn lod_filtering_keeps_original_chunk_materials() {
        let materials: Vec<String> = [
            "cards", "cap", "cards", "cap", "cards", "cap", "cards", "cap",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        assert_eq!(material_for_chunk(&materials, 0), "cards");
        assert_eq!(material_for_chunk(&materials, 1), "cap");
        assert_eq!(material_for_chunk(&materials, 5), "cap");
        assert_eq!(material_for_chunk(&materials[..2], 5), "cap");
    }

    #[test]
    fn levels_apply_multiply_then_add_and_clamp() {
        assert!((apply_levels(0.5, [2.0, -0.4]) - 0.6).abs() < 1e-6);
        assert!((apply_levels(1.0, [2.0, 0.0]) - 1.0).abs() < f32::EPSILON);
        assert!(apply_levels(0.0, [1.0, -1.0]).abs() < f32::EPSILON);
    }
}
