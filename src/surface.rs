//! Archive-backed queries against native static world collision surfaces.
use crate::{
    assets::{AssetReader, DocumentCache, ResourceKey},
    material::ArchiveSet,
    schema::RedSchema,
    world::{WorldError, collision_groups},
};
use glam::{DQuat, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};

/// A vertical query bounded around an intended world level, in metres.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SurfaceQuery {
    pub position: [f64; 3],
    pub above: f64,
    pub below: f64,
}
/// Physical surface hit, with an upward normal and native resource provenance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SurfaceHit {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub sector: String,
    /// Height of the next physical surface above this level, if present in the query.
    pub overhead_m: Option<f64>,
}
/// Results stay in query order. Missing geometry is explicit rather than extrapolated.
#[derive(Debug, Serialize, Deserialize)]
pub struct SurfaceSurvey {
    pub hits: Vec<Option<SurfaceHit>>,
    pub sectors: usize,
    pub triangles: usize,
    pub diagnostics: Vec<String>,
}
struct Triangle {
    points: [DVec3; 3],
    normal: DVec3,
    sector: usize,
}
fn cell(p: DVec3) -> (i32, i32) {
    ((p.x / 16.).floor() as i32, (p.y / 16.).floor() as i32)
}
fn root(v: &Value) -> &Value {
    &v["Data"]["RootChunk"]
}
fn scalar(v: &Value) -> Option<f64> {
    v.get("$value").unwrap_or(v).as_f64()
}
fn vector(v: &Value) -> Option<DVec3> {
    Some(DVec3::new(
        scalar(&v["X"])?,
        scalar(&v["Y"])?,
        scalar(&v["Z"])?,
    ))
}
fn add(
    mut points: [DVec3; 3],
    sector: usize,
    mirrored: bool,
    wanted: &HashSet<(i32, i32)>,
    triangles: &mut Vec<Triangle>,
    cells: &mut HashMap<(i32, i32), Vec<usize>>,
) {
    if mirrored {
        points.swap(1, 2);
    }
    let normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize_or_zero();
    if normal.z < 0.45 {
        return;
    }
    let low = cell(points[0].min(points[1]).min(points[2]));
    let high = cell(points[0].max(points[1]).max(points[2]));
    let mut used = false;
    // Query cells bound work even for unusually large world triangles.
    let mut insert = |c| {
        if wanted.contains(&c) {
            cells.entry(c).or_default().push(triangles.len());
            used = true;
        }
    };
    if i64::from(high.0 - low.0 + 1) * i64::from(high.1 - low.1 + 1) < 4096 {
        for x in low.0..=high.0 {
            for y in low.1..=high.1 {
                insert((x, y));
            }
        }
    } else {
        for &c in wanted {
            if c.0 >= low.0 && c.0 <= high.0 && c.1 >= low.1 && c.1 <= high.1 {
                insert(c);
            }
        }
    }
    if used {
        triangles.push(Triangle {
            points,
            normal,
            sector,
        });
    }
}
fn height(t: &Triangle, p: DVec3) -> Option<f64> {
    let [a, b, c] = t.points;
    let u = (b - a).truncate();
    let v = (c - a).truncate();
    let w = (p - a).truncate();
    let determinant = u.perp_dot(v);
    if determinant.abs() < 1e-10 {
        return None;
    }
    let s = w.perp_dot(v) / determinant;
    let r = u.perp_dot(w) / determinant;
    if s < -1e-7 || r < -1e-7 || s + r > 1.0000001 {
        return None;
    }
    Some(a.z + s * (b.z - a.z) + r * (c.z - a.z))
}

/// Samples the nearest physical surface on the intended level from installed archives.
/// Reads only sector descriptors intersecting the queries; collision meshes share one cache.
///
/// # Errors
/// Rejects invalid queries or unreadable world metadata. Individual unsupported shapes
/// are reported in `diagnostics`; uncovered queries remain `None`.
pub fn sample(
    archives: &ArchiveSet,
    schema: &RedSchema,
    queries: &[SurfaceQuery],
) -> Result<SurfaceSurvey, WorldError> {
    let bad = |e: String| WorldError::Invalid(e);
    if queries.iter().any(|q| {
        !q.position.iter().all(|n| n.is_finite())
            || !q.above.is_finite()
            || !q.below.is_finite()
            || q.above < 0.
            || q.below < 0.
    }) {
        return Err(bad("invalid surface query".into()));
    }
    let cache = DocumentCache::default();
    let mut query_bounds = HashMap::<(i32, i32), (f64, f64)>::new();
    for q in queries {
        let range = query_bounds
            .entry(cell(DVec3::from_array(q.position)))
            .or_insert((f64::INFINITY, f64::NEG_INFINITY));
        range.0 = range.0.min(q.position[2] - q.below);
        range.1 = range.1.max(q.position[2] + q.above);
    }
    let mut sectors = BTreeSet::new();
    for path in [
        r"base\worlds\03_night_city\_compiled\default\blocks\all.streamingblock",
        r"base\worlds\03_night_city\_compiled\default\ep1\blocks\all.streamingblock",
    ] {
        let key = ResourceKey::parse(path);
        let mut reader = AssetReader::with_cache(archives, schema, &cache);
        if !reader.resource_exists(&key) {
            continue;
        }
        let document = reader.document(&key).map_err(|e| bad(e.to_string()))?;
        for d in root(&document)["descriptors"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(path) = d["data"]["DepotPath"]["$value"].as_str()
                && path.ends_with(".streamingsector")
                && query_bounds.iter().any(|(&(x, y), &(low, high))| {
                    let bounds = &d["streamingBox"];
                    let (Some(a), Some(b)) = (vector(&bounds["Min"]), vector(&bounds["Max"]))
                    else {
                        return false;
                    };
                    a.x <= f64::from(x + 1) * 16.
                        && b.x >= f64::from(x) * 16.
                        && a.y <= f64::from(y + 1) * 16.
                        && b.y >= f64::from(y) * 16.
                        && a.z <= high
                        && b.z >= low
                })
            {
                sectors.insert(path.to_owned());
            }
        }
    }
    let sources = sectors.into_iter().collect::<Vec<_>>();
    let wanted = queries
        .iter()
        .map(|q| cell(DVec3::from_array(q.position)))
        .collect::<HashSet<_>>();
    let mut triangles = Vec::new();
    let mut cells = HashMap::new();
    let mut diagnostics = Vec::new();
    let decoder = crate::codec::SelectedExportDecoder::new(schema);
    for (source, path) in sources.iter().enumerate() {
        let reader = AssetReader::with_cache(archives, schema, &cache);
        let bytes = reader
            .read_bytes(&ResourceKey::parse(path))
            .map_err(|e| bad(e.to_string()))?;
        let nodes = decoder
            .decode(
                &bytes,
                &BTreeSet::from([
                    "worldCollisionNode".into(),
                    "worldCookedCollisionNode".into(),
                    "worldTerrainCollisionNode".into(),
                ]),
            )
            .map_err(|e| bad(format!("{path}: {e}")))?;
        for n in &nodes {
            if n["$type"] == "worldTerrainCollisionNode" {
                let terrain = crate::world::terrain_collision(n)?;
                for face in &terrain.triangles {
                    add(
                        face.map(|i| {
                            DVec3::from_array(terrain.vertices[i as usize].map(f64::from))
                        }),
                        source,
                        false,
                        &wanted,
                        &mut triangles,
                        &mut cells,
                    );
                }
                continue;
            }
            for group in collision_groups(n)? {
                let sector = group.sector_hash;
                for shape in group.shapes {
                    let p = DVec3::from_array(shape.position.map(f64::from));
                    let rotation = DQuat::from_array(shape.rotation.map(f64::from));
                    let scale = DVec3::from_array(shape.scale.map(f64::from));
                    let transform =
                        |v: [f32; 3]| p + rotation * (DVec3::from_array(v.map(f64::from)) * scale);
                    if let Some(extents) = shape.half_extents {
                        let vertices = (0..8)
                            .map(|i| {
                                transform(std::array::from_fn(|a| {
                                    extents[a] * if i & (1 << a) == 0 { -1. } else { 1. }
                                }))
                            })
                            .collect::<Vec<_>>();
                        for face in [
                            [0, 3, 1],
                            [0, 2, 3],
                            [4, 7, 6],
                            [4, 5, 7],
                            [0, 5, 4],
                            [0, 1, 5],
                            [2, 7, 3],
                            [2, 6, 7],
                            [0, 6, 2],
                            [0, 4, 6],
                            [1, 7, 5],
                            [1, 3, 7],
                        ] {
                            add(
                                face.map(|i| vertices[i]),
                                source,
                                scale.x * scale.y * scale.z < 0.,
                                &wanted,
                                &mut triangles,
                                &mut cells,
                            );
                        }
                    } else if let Some(hash) = shape.geometry {
                        match reader.collision_geometry(sector, hash) {
                            Ok(geometry) => {
                                for face in &geometry.triangles {
                                    add(
                                        face.map(|i| transform(geometry.vertices[i as usize])),
                                        source,
                                        scale.x * scale.y * scale.z < 0.,
                                        &wanted,
                                        &mut triangles,
                                        &mut cells,
                                    );
                                }
                            }
                            Err(e) => {
                                if diagnostics.len() < 64 {
                                    diagnostics.push(format!(
                                        "{path} shape {}/{}: {e}",
                                        shape.actor, shape.shape
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let hits = queries
        .par_iter()
        .map(|q| {
            let p = DVec3::from_array(q.position);
            let mut candidates = cells
                .get(&cell(p))
                .into_iter()
                .flatten()
                .filter_map(|i| {
                    let t = &triangles[*i];
                    let z = height(t, p)?;
                    (z >= p.z - q.below && z <= p.z + q.above).then_some((z, t))
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|(a, _), (b, _)| a.total_cmp(b));
            // Thin road decks and underlying terrain belong to one surface level.
            // Keep bridges/underpasses separate, then use the top of the nearest level.
            let mut groups = Vec::new();
            let mut start = 0;
            for i in 1..=candidates.len() {
                // Half-metre road layers need a small margin for cooked mesh
                // quantization and the vertical thickness of sloping decks.
                if i == candidates.len() || candidates[i].0 - candidates[i - 1].0 > 0.51 {
                    groups.push(start..i);
                    start = i;
                }
            }
            groups
                .into_iter()
                .min_by(|a, b| {
                    let distance = |range: &std::ops::Range<usize>| {
                        candidates[range.clone()]
                            .iter()
                            .map(|(z, _)| (z - p.z).abs())
                            .fold(f64::INFINITY, f64::min)
                    };
                    distance(a).total_cmp(&distance(b))
                })
                .and_then(|range| {
                    let (z, t) = *candidates.get(range.end.checked_sub(1)?)?;
                    let overhead = candidates.get(range.end).map(|(next, _)| next - z);
                    Some(SurfaceHit {
                        position: [p.x, p.y, z],
                        normal: t.normal.to_array(),
                        sector: sources[t.sector].clone(),
                        overhead_m: overhead,
                    })
                })
        })
        .collect();
    Ok(SurfaceSurvey {
        hits,
        sectors: sources.len(),
        triangles: triangles.len(),
        diagnostics,
    })
}
