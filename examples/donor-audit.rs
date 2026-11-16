//! Read-only replacement-donor audit using native archive dependency tables and TweakDB.
use anyhow::{Context, Result, ensure};
use ghostline_red::{
    archive::{depot_path_hash, read_archive},
    tweakdb::TweakDb,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

fn archives(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            archives(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "archive") {
            out.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5,
        "usage: donor-audit GAME PATH_DICTIONARY CANDIDATES OUTPUT"
    );
    let game = Path::new(&args[1]);
    let names: HashMap<_, _> = fs::read_to_string(&args[2])?
        .lines()
        .map(|p| (depot_path_hash(p), p.to_owned()))
        .collect();
    let candidates: Vec<_> = fs::read_to_string(&args[3])?
        .lines()
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect();
    let targets: HashMap<_, _> = candidates
        .iter()
        .map(|p| (depot_path_hash(p), p.clone()))
        .collect();
    let mut found: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut inbound: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut paths = Vec::new();
    for folder in ["archive/pc/content", "archive/pc/ep1", "archive/pc/hot"] {
        let root = game.join(folder);
        if root.exists() {
            archives(&root, &mut paths)?;
        }
    }
    paths.sort();
    let mut entries = 0;
    let mut dependencies = 0;
    let mut import_scanned = 0;
    let mut import_errors = Vec::new();
    let mut imports: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for path in &paths {
        let index = read_archive(path).with_context(|| path.display().to_string())?;
        entries += index.entries.len();
        dependencies += index.dependency_count as usize;
        let offset = index.header.index_position
            + 28
            + index.entries.len() as u64 * 56
            + u64::from(index.segment_count) * 16;
        let mut file = fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; index.dependency_count as usize * 8];
        file.read_exact(&mut bytes)?;
        let deps: Vec<_> = bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight-byte chunk")))
            .collect();
        let selected: Vec<_> = index
            .entries
            .iter()
            .filter(|entry| {
                names.get(&entry.name_hash).is_some_and(|name| {
                    [
                        ".ent",
                        ".app",
                        ".quest",
                        ".questphase",
                        ".scene",
                        ".streamingsector",
                        ".streamingblock",
                        ".community",
                        ".csv",
                    ]
                    .iter()
                    .any(|ext| name.ends_with(ext))
                })
            })
            .collect();
        let inspected: Vec<_> = selected
            .par_iter()
            .map_init(
                || fs::File::open(path),
                |file, entry| -> Result<_> {
                    let file = file
                        .as_mut()
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                    let segment = index
                        .segments
                        .get(entry.segments_start as usize)
                        .context("missing first segment")?;
                    file.seek(SeekFrom::Start(segment.offset))?;
                    let mut data = vec![0; segment.compressed_size as usize];
                    file.read_exact(&mut data)?;
                    if data.starts_with(b"KARK") {
                        let size =
                            u32::from_le_bytes(data.get(4..8).context("KARK size")?.try_into()?);
                        data = ghostline_red::kraken::decode(&data[8..], size as usize)?;
                    }
                    let resource = &names[&entry.name_hash];
                    if !data.starts_with(b"CR2W") {
                        return Ok(Vec::new());
                    }
                    let inspected = ghostline_red::cr2w::inspect_bytes(&data)
                        .with_context(|| resource.clone())?;
                    Ok(inspected
                        .imports
                        .into_iter()
                        .filter_map(|import| {
                            targets
                                .get(&depot_path_hash(&import.depot_path))
                                .map(|target| {
                                    (target.clone(), json!({"archive":path,"resource":resource}))
                                })
                        })
                        .collect::<Vec<_>>())
                },
            )
            .collect();
        import_scanned += selected.len();
        for result in inspected {
            match result {
                Ok(hits) => {
                    for (target, hit) in hits {
                        imports.entry(target).or_default().push(hit);
                    }
                }
                Err(error) => import_errors.push(format!("{error:#}")),
            }
        }
        for entry in &index.entries {
            if let Some(target) = targets.get(&entry.name_hash) {
                found
                    .entry(target.clone())
                    .or_default()
                    .push(json!({"archive":path,"bytes":entry.size}));
            }
            let refs = deps
                .get(entry.dependencies_start as usize..entry.dependencies_end as usize)
                .context("invalid archive dependency range")?;
            for hash in refs {
                if let Some(target) = targets.get(hash) {
                    inbound.entry(target.clone()).or_default().push(json!({"archive":path,"resource":names.get(&entry.name_hash),"hash":entry.name_hash.to_string()}));
                }
            }
        }
    }
    let mut tweaks: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut resource_flats: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for cache in ["tweakdb.bin", "tweakdb_ep1.bin"] {
        let path = game.join("r6/cache").join(cache);
        let db = TweakDb::from_bytes(&fs::read(&path)?)?;
        for record in db.records.keys() {
            if let Some(value) = db.property(*record, "entityTemplatePath") {
                if let Some(text) = value.value.as_str()
                    && let Some(target) =
                        targets.get(&text.parse().unwrap_or_else(|_| depot_path_hash(text)))
                {
                    tweaks.entry(target.clone()).or_default().push(json!({"cache":cache,"record":record.to_string(),"entityTemplatePath":value,"appearanceName":db.property(*record,"appearanceName")}));
                }
            }
        }
        for (flat, value) in &db.flats {
            if !value.native_type.contains("raRef:") {
                continue;
            }
            let values = value
                .value
                .as_array()
                .map_or_else(|| vec![&value.value], |items| items.iter().collect());
            for value in values {
                if let Some(text) = value.as_str()
                    && let Some(target) =
                        targets.get(&text.parse().unwrap_or_else(|_| depot_path_hash(text)))
                {
                    resource_flats
                        .entry(target.clone())
                        .or_default()
                        .push(json!({"cache":cache,"flat":flat.to_string()}));
                }
            }
        }
    }
    let rows: Vec<_> = candidates.iter().map(|target| json!({"path":target,"hash":depot_path_hash(target).to_string(),"present":found.get(target),"archive_references":inbound.get(target),"native_import_references":imports.get(target),"character_records":tweaks.get(target),"tweak_resource_flats":resource_flats.get(target)})).collect();
    let output = json!({"archives":paths.len(),"entries":entries,"dependencies":dependencies,"import_resources_scanned":import_scanned,"import_errors":import_errors,"scope":"vanilla archive index dependencies, known entity/appearance/quest/scene/world/community/CSV CR2W import tables and TweakDB resource-reference flats; excludes scripts, opaque embedded references and runtime verification","candidates":rows});
    fs::write(&args[4], serde_json::to_vec_pretty(&output)?)?;
    println!(
        "Audited {} archives, {entries} entries, {dependencies} dependency edges; report {}",
        paths.len(),
        args[4]
    );
    Ok(())
}
