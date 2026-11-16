//! Native creator morph transfer without interchange files.
use crate::{
    assets::AssetError,
    mesh::MeshChunk,
    morph::{MorphChunk, MorphTarget},
};
use glam::Vec3;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
fn invalid(message: &str) -> AssetError {
    AssetError::Invalid(message.into())
}

struct Tree {
    index: usize,
    axis: usize,
    left: Option<Box<Tree>>,
    right: Option<Box<Tree>>,
}
impl Tree {
    fn build(mut indices: Vec<usize>, points: &[[f32; 3]], depth: usize) -> Option<Box<Self>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 3;
        indices.sort_unstable_by(|&a, &b| points[a][axis].total_cmp(&points[b][axis]));
        let middle = indices.len() / 2;
        Some(Box::new(Self {
            index: indices[middle],
            axis,
            left: Self::build(indices[..middle].to_vec(), points, depth + 1),
            right: Self::build(indices[middle + 1..].to_vec(), points, depth + 1),
        }))
    }
    fn nearest(&self, point: Vec3, points: &[[f32; 3]], best: &mut Vec<(f32, usize)>) {
        let distance = point.distance_squared(points[self.index].into());
        best.push((distance, self.index));
        best.sort_by(|a, b| a.0.total_cmp(&b.0));
        best.truncate(4);
        let delta = point[self.axis] - points[self.index][self.axis];
        let (near, far) = if delta < 0. {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        if let Some(node) = near {
            node.nearest(point, points, best)
        }
        if (best.len() < 4 || delta * delta <= best.last().map_or(f32::INFINITY, |b| b.0))
            && let Some(node) = far
        {
            node.nearest(point, points, best)
        }
    }
}
#[derive(Clone, Copy)]
struct Distance(f32, usize);
impl PartialEq for Distance {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for Distance {}
impl PartialOrd for Distance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Distance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .total_cmp(&self.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}
fn key(point: [f32; 3]) -> [i64; 3] {
    point.map(|v| (v as f64 * 100000.).round() as i64)
}

fn neck_weights(chunk: &MeshChunk, neck_y: f32) -> Result<Vec<f32>, AssetError> {
    let mut points = Vec::<Vec3>::new();
    let mut lookup = HashMap::new();
    let mut remap = Vec::new();
    for &point in &chunk.positions {
        let next = points.len();
        let i = *lookup.entry(key(point)).or_insert_with(|| {
            points.push(point.into());
            next
        });
        remap.push(i)
    }
    let mut edges = HashMap::new();
    for tri in chunk.indices.chunks_exact(3) {
        let [a, b, c] = [
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        ];
        for (a, b) in [(a, b), (b, c), (c, a)] {
            if a != b {
                *edges.entry((a.min(b), a.max(b))).or_insert(0usize) += 1;
            }
        }
    }
    let mut graph = vec![Vec::new(); points.len()];
    let mut boundary = vec![Vec::new(); points.len()];
    for ((a, b), count) in edges {
        graph[a].push(b);
        graph[b].push(a);
        if count == 1 {
            boundary[a].push(b);
            boundary[b].push(a)
        }
    }
    let mut seen = vec![false; points.len()];
    let mut collars = Vec::new();
    for i in 0..points.len() {
        if seen[i] || boundary[i].is_empty() {
            continue;
        }
        let mut stack = vec![i];
        seen[i] = true;
        let mut group = Vec::new();
        while let Some(j) = stack.pop() {
            group.push(j);
            for &k in &boundary[j] {
                if !seen[k] {
                    seen[k] = true;
                    stack.push(k)
                }
            }
        }
        if group.len() >= 8 {
            let height = group.iter().map(|&j| points[j].y).sum::<f32>() / group.len() as f32;
            if height <= neck_y + 0.03 {
                collars.push((height, group))
            }
        }
    }
    collars.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some((_, collar)) = collars.first() else {
        return Err(invalid("native head has no identifiable open neck collar"));
    };
    let mut distances = vec![f32::INFINITY; points.len()];
    let mut queue = BinaryHeap::new();
    for &i in collar {
        distances[i] = 0.;
        queue.push(Distance(0., i))
    }
    while let Some(Distance(d, i)) = queue.pop() {
        if d > distances[i] {
            continue;
        }
        for &j in &graph[i] {
            let candidate = d + points[i].distance(points[j]);
            if candidate < distances[j] {
                distances[j] = candidate;
                queue.push(Distance(candidate, j));
            }
        }
    }
    Ok(remap
        .into_iter()
        .map(|i| {
            let t = ((distances[i] - 0.005) / 0.020).clamp(0., 1.);
            t * t * (3. - 2. * t)
        })
        .collect())
}

/// Transfers sparse native facial targets onto another native mesh and every LOD.
/// UV correspondence is disambiguated spatially; missing UV matches use four
/// nearest normalized positions. A supplied neck height pins the open collar
/// and a 5 mm band, reaching full influence after 25 mm along the surface.
///
/// # Errors
/// Returns an error for missing geometry, invalid indices or an unidentifiable collar.
pub fn transfer(
    source: &[MeshChunk],
    targets: &[MorphTarget],
    donor: &[MeshChunk],
    neck_y: Option<f32>,
) -> Result<Vec<MorphTarget>, AssetError> {
    if neck_y.is_some_and(|v| !v.is_finite()) {
        return Err(invalid("neck anchor must be finite"));
    }
    for chunk in source.iter().chain(donor) {
        if chunk.positions.iter().flatten().any(|v| !v.is_finite())
            || chunk.tex_coords.iter().flatten().any(|v| !v.is_finite())
            || chunk.indices.len() % 3 != 0
            || chunk
                .indices
                .iter()
                .any(|&i| i as usize >= chunk.positions.len())
            || (!chunk.tex_coords.is_empty() && chunk.tex_coords.len() != chunk.positions.len())
        {
            return Err(invalid("invalid morph transfer geometry"));
        }
    }
    for target in targets {
        for chunk in &target.chunks {
            if chunk.vertices.len() != chunk.positions.len()
                || chunk.positions.iter().flatten().any(|v| !v.is_finite())
            {
                return Err(invalid("invalid source morph deltas"));
            }
        }
    }
    let primary = source
        .iter()
        .max_by_key(|c| c.positions.len())
        .ok_or_else(|| invalid("morph transfer source is empty"))?;
    if primary.positions.is_empty() {
        return Err(invalid("morph source has no vertices"));
    }
    let mut source_min = Vec3::splat(f32::INFINITY);
    let mut source_max = Vec3::splat(f32::NEG_INFINITY);
    for &p in &primary.positions {
        source_min = source_min.min(p.into());
        source_max = source_max.max(p.into())
    }
    let mut uv = HashMap::<[i64; 2], Vec<usize>>::new();
    for (i, p) in primary.tex_coords.iter().enumerate() {
        uv.entry(p.map(|v| (v as f64 * 100000.).round() as i64))
            .or_default()
            .push(i);
    }
    let tree = Tree::build(
        (0..primary.positions.len()).collect(),
        &primary.positions,
        0,
    )
    .ok_or_else(|| invalid("empty transfer tree"))?;
    let mut output: Vec<_> = targets
        .iter()
        .map(|t| MorphTarget {
            name: t.name.clone(),
            region: t.region.clone(),
            chunks: Vec::new(),
        })
        .collect();
    let mut dense = BTreeMap::new();
    for target in targets {
        let mut values = vec![Vec3::ZERO; primary.positions.len()];
        for change in target
            .chunks
            .iter()
            .filter(|c| c.chunk == primary.chunk_index)
        {
            for (&vertex, &delta) in change.vertices.iter().zip(&change.positions) {
                let Some(v) = values.get_mut(vertex as usize) else {
                    return Err(invalid("source morph vertex is out of bounds"));
                };
                *v = delta.into();
            }
        }
        dense.insert(target.name.as_str(), values);
    }
    for chunk in donor {
        if chunk.positions.is_empty() {
            continue;
        }
        let mut minimum = Vec3::splat(f32::INFINITY);
        let mut maximum = Vec3::splat(f32::NEG_INFINITY);
        for &p in &chunk.positions {
            minimum = minimum.min(p.into());
            maximum = maximum.max(p.into())
        }
        let weights = if let Some(y) = neck_y {
            neck_weights(chunk, y)?
        } else {
            vec![1.; chunk.positions.len()]
        };
        let mapping: Vec<Vec<(usize, f32)>> = chunk
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let point = source_min
                    + (Vec3::from(p) - minimum) / (maximum - minimum).max(Vec3::splat(1e-9))
                        * (source_max - source_min);
                if let Some(indices) = chunk
                    .tex_coords
                    .get(i)
                    .and_then(|p| uv.get(&p.map(|v| (v as f64 * 100000.).round() as i64)))
                {
                    let nearest = *indices
                        .iter()
                        .min_by(|&&a, &&b| {
                            point
                                .distance_squared(primary.positions[a].into())
                                .total_cmp(&point.distance_squared(primary.positions[b].into()))
                        })
                        .expect("nonempty UV group");
                    return vec![(nearest, 1.)];
                }
                let mut best = Vec::new();
                tree.nearest(point, &primary.positions, &mut best);
                let sum = best.iter().map(|(d, _)| 1. / d.max(1e-14)).sum::<f32>();
                best.into_iter()
                    .map(|(d, i)| (i, (1. / d.max(1e-14)) / sum))
                    .collect()
            })
            .collect();
        for target in &mut output {
            let mut change = MorphChunk {
                chunk: chunk.chunk_index,
                vertices: Vec::new(),
                positions: Vec::new(),
                normals: Vec::new(),
                tangents: Vec::new(),
            };
            for (i, row) in mapping.iter().enumerate() {
                let delta = row
                    .iter()
                    .map(|&(j, w)| dense[target.name.as_str()][j] * w)
                    .sum::<Vec3>()
                    * weights[i];
                if delta.length_squared() > 1e-18 {
                    change.vertices.push(
                        u16::try_from(i).map_err(|_| {
                            invalid("donor chunk exceeds native vertex index range")
                        })?,
                    );
                    change.positions.push(delta.to_array());
                    change.normals.push([0.; 3]);
                    change.tangents.push([0.; 3]);
                }
            }
            target.chunks.push(change);
        }
    }
    Ok(output)
}
