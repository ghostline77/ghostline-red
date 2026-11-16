//! Conservative cross-mod character-slot audit. Reads game data; never writes into the game.
use aho_corasick::AhoCorasick;
use anyhow::{Context, Result, ensure};
use ghostline_red::{
    archive::{self, ArchiveIndex},
    codec, cr2w, kraken,
    schema::RedSchema,
};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

fn walk(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn normalized(path: &str) -> String {
    path.replace('/', "\\")
        .split('\\')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\\")
        .to_ascii_lowercase()
}

struct Matcher {
    search: AhoCorasick,
    patterns: Vec<(Vec<u8>, usize, &'static str)>,
    stems: AhoCorasick,
    stem_names: Vec<String>,
    targets: HashMap<u64, usize>,
    stem_targets: HashMap<String, Vec<usize>>,
}

impl Matcher {
    fn new(candidates: &[String]) -> Result<Self> {
        let mut patterns = Vec::new();
        let mut stems = BTreeSet::new();
        let mut stem_targets: HashMap<String, Vec<usize>> = HashMap::new();
        let mut targets = HashMap::new();
        for (id, candidate) in candidates.iter().enumerate() {
            let path = normalized(candidate);
            let hash = archive::depot_path_hash(&path);
            targets.insert(hash, id);
            patterns.push((hash.to_le_bytes().to_vec(), id, "resource_hash"));
            patterns.push((hash.to_be_bytes().to_vec(), id, "resource_hash"));
            for variant in [
                path.clone(),
                path.replace('\\', "/"),
                path.replace('\\', "\\\\"),
            ] {
                let wide: Vec<_> = variant.encode_utf16().flat_map(u16::to_le_bytes).collect();
                patterns.push((variant.into_bytes(), id, "resource_path"));
                patterns.push((wide, id, "resource_path"));
            }
            let stem = path
                .rsplit('\\')
                .next()
                .context("candidate filename")?
                .rsplit_once('.')
                .context("candidate extension")?
                .0
                .to_owned();
            stems.insert(stem.clone());
            stem_targets.entry(stem).or_default().push(id);
        }
        let stem_names: Vec<_> = stems.into_iter().collect();
        Ok(Self {
            search: AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build(patterns.iter().map(|(bytes, _, _)| bytes))?,
            patterns,
            stems: AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build(&stem_names)?,
            stem_names,
            targets,
            stem_targets,
        })
    }

    fn hits(&self, bytes: &[u8]) -> BTreeMap<usize, BTreeSet<&'static str>> {
        let mut hits: BTreeMap<usize, BTreeSet<_>> = BTreeMap::new();
        for found in self.search.find_overlapping_iter(bytes) {
            let (pattern, id, kind) = &self.patterns[found.pattern().as_usize()];
            if *kind == "resource_hash" && &bytes[found.start()..found.end()] != pattern {
                continue;
            }
            hits.entry(*id).or_default().insert(*kind);
        }
        hits
    }
}

fn segment(file: &mut File, segment: archive::ArchiveSegment) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(segment.offset))?;
    let mut bytes = vec![0; segment.compressed_size as usize];
    file.read_exact(&mut bytes)?;
    if bytes.starts_with(b"KARK") && segment.compressed_size != segment.size {
        let length = u32::from_le_bytes(bytes.get(4..8).context("KARK header")?.try_into()?);
        return Ok(kraken::decode(&bytes[8..], length as usize)?);
    }
    Ok(bytes)
}

fn full_payload(
    file: &mut File,
    index: &ArchiveIndex,
    entry: &archive::ArchiveEntry,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for part in &index.segments[entry.segments_start as usize..entry.segments_end as usize] {
        bytes.extend(segment(file, *part)?);
    }
    Ok(bytes)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--verify-matcher" {
        let matcher = Matcher::new(&[args[3].clone()])?;
        let found = matcher.hits(&fs::read(&args[2])?);
        ensure!(
            found.contains_key(&0),
            "native reference positive control was not found"
        );
        println!("Native reference positive control passed: {:?}", found[&0]);
        return Ok(());
    }
    ensure!(
        args.len() == 5,
        "usage: donor-deep-audit GAME PRIOR_REPORT PATH_DICTIONARY OUTPUT"
    );
    let game = Path::new(&args[1]);
    let prior: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let candidates: Vec<String> = prior["candidates"]
        .as_array()
        .context("prior candidates")?
        .iter()
        .filter(|candidate| {
            let path = candidate["path"].as_str().unwrap_or_default();
            let character = (path.starts_with("base\\characters\\entities\\")
                || path.starts_with("ep1\\characters\\entities\\"))
                && path.ends_with(".ent")
                || (path.starts_with("base\\characters\\appearances\\")
                    || path.starts_with("ep1\\characters\\appearances\\"))
                    && path.ends_with(".app");
            character
                && candidate["present"].is_array()
                && [
                    "native_import_references",
                    "archive_references",
                    "tweak_resource_flats",
                    "character_records",
                ]
                .iter()
                .all(|key| candidate[key].is_null())
        })
        .map(|candidate| candidate["path"].as_str().unwrap().to_owned())
        .collect();
    let matcher = Matcher::new(&candidates)?;
    let names: HashMap<_, _> = fs::read_to_string(&args[3])?
        .lines()
        .map(|name| (archive::depot_path_hash(name), name.to_owned()))
        .collect();
    let schema = RedSchema::load(Some(game), None)?;
    let mut archives = Vec::new();
    for root in ["archive/pc/content", "archive/pc/ep1", "archive/pc/hot"] {
        walk(&game.join(root), &mut archives)?;
    }
    archives.retain(|path| path.extension().is_some_and(|ext| ext == "archive"));
    archives.sort();
    let mut hits: BTreeMap<usize, Vec<Value>> = BTreeMap::new();
    let mut details: BTreeMap<usize, Value> = BTreeMap::new();
    let mut errors = Vec::new();
    let mut skipped_media = 0;
    let mut scanned_resources = 0;
    let mut scanned_bytes: u64 = 0;
    let mut factory_tables = 0;
    let mut factory_rows = Vec::new();
    let media = [
        "wem", "opuspak", "opusinfo", "bk2", "bnk", "bnkpack", "ogg", "wav",
    ];
    for path in &archives {
        let index = archive::read_archive(path)?;
        let entries: Vec<_> = index
            .entries
            .iter()
            .filter(|entry| {
                let excluded = names
                    .get(&entry.name_hash)
                    .and_then(|name| name.rsplit('.').next())
                    .is_some_and(|ext| media.contains(&ext));
                if excluded {
                    skipped_media += 1;
                }
                !excluded
            })
            .collect();
        let results: Vec<_> = entries.par_iter().map_init(|| File::open(path), |file, entry| -> Result<_> {
            let file = file.as_mut().map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let name = names.get(&entry.name_hash).cloned().unwrap_or_else(|| format!("hash:{:016x}", entry.name_hash));
            let first = *index.segments.get(entry.segments_start as usize).context("first segment")?;
            let main = segment(file, first).with_context(|| name.clone())?;
            let mut matches = matcher.hits(&main);
            let mut bytes = main.len() as u64;
            let mut rows = Vec::new();
            let mut detail = None;
            let mut table = false;
            if main.starts_with(b"CR2W") {
                let inspection = cr2w::inspect_bytes(&main).with_context(|| name.clone())?;
                let root_type = inspection.exports.first().map(|export| export.class_name.as_str()).unwrap_or_default();
                let structural = matches!(root_type, "entEntityTemplate" | "appearanceAppearanceResource" | "C2dArray")
                    || [".quest", ".questphase", ".scene", ".streamingsector", ".streamingblock", ".community", ".behavior", ".animgraph", ".scenerid"].iter().any(|ext| name.ends_with(ext));
                if structural {
                    for part in &index.segments[entry.segments_start as usize + 1..entry.segments_end as usize] {
                        let data = segment(file, *part).with_context(|| name.clone())?;
                        bytes += data.len() as u64;
                        for (id, kinds) in matcher.hits(&data) { matches.entry(id).or_default().extend(kinds); }
                    }
                }
                let candidate = matcher.targets.get(&entry.name_hash).copied();
                if root_type == "C2dArray" || candidate.is_some() {
                    let full = full_payload(file, &index, entry)?;
                    let document = codec::decode_wkit_from_bytes(&full, &schema).with_context(|| name.clone())?;
                    let root = &document["Data"]["RootChunk"];
                    if root_type == "C2dArray" {
                        table = true;
                        if let Some(data) = root["data"].as_array() {
                            for (row_index, row) in data.iter().enumerate() {
                                let encoded = serde_json::to_vec(row)?;
                                let found = matcher.hits(&encoded);
                                for (id, kinds) in found {
                                    matches.entry(id).or_default().extend(kinds);
                                    rows.push(json!({"candidate":candidates[id],"table":name,"row_index":row_index,"headers":root["headers"],"row":row}));
                                }
                            }
                        }
                    }
                    if let Some(id) = candidate {
                        let chunks = &root["compiledData"]["Data"]["Chunks"];
                        let puppet = chunks.as_array().and_then(|chunks| chunks.first()).and_then(|chunk| chunk["$type"].as_str());
                        let mut definition_names = Vec::new();
                        if let Some(appearances) = root["appearances"].as_array() {
                            for appearance in appearances {
                                let definition = appearance.get("Data").unwrap_or(appearance);
                                definition_names.push(definition["name"]["$value"].as_str().unwrap_or_default().to_owned());
                            }
                        }
                        detail = Some((id, json!({"root_type":root_type,"puppet_type":puppet,"appearance_names":definition_names,"compiled_chunks":chunks.as_array().map(Vec::len),"archive":path})));
                    }
                }
            }
            matches.remove(&matcher.targets.get(&entry.name_hash).copied().unwrap_or(usize::MAX));
            Ok((name, matches, bytes, rows, table, detail))
        }).collect();
        for result in results {
            match result {
                Ok((resource, found, bytes, rows, table, detail)) => {
                    scanned_resources += 1;
                    scanned_bytes += bytes;
                    factory_tables += usize::from(table);
                    factory_rows.extend(rows);
                    if let Some((id, detail)) = detail {
                        details.insert(id, detail);
                    }
                    for (id, kinds) in found {
                        hits.entry(id)
                            .or_default()
                            .push(json!({"archive":path,"resource":resource,"kinds":kinds}));
                    }
                }
                Err(error) => errors.push(format!("{error:#}")),
            }
        }
        eprintln!(
            "{}: {scanned_resources} resources, {factory_tables} tables, {} errors",
            path.file_name().unwrap().to_string_lossy(),
            errors.len()
        );
    }
    let mut loose = Vec::new();
    for root in [
        "r6/scripts",
        "r6/config",
        "engine/config",
        "red4ext/plugins",
    ] {
        walk(&game.join(root), &mut loose)?;
    }
    loose.retain(|path| {
        path.extension().is_some_and(|ext| {
            ["reds", "ini", "json", "xml", "csv", "yaml", "xl"]
                .iter()
                .any(|kind| ext == *kind)
        })
    });
    for name in [
        "r6/cache/final.redscripts",
        "r6/cache/final.redscripts.modded",
        "bin/x64/Cyberpunk2077.exe",
    ] {
        let path = game.join(name);
        if path.exists() {
            loose.push(path);
        }
    }
    let mut script_name_hints: BTreeMap<usize, Vec<Value>> = BTreeMap::new();
    for path in &loose {
        let data = fs::read(path)?;
        for (id, kinds) in matcher.hits(&data) {
            hits.entry(id)
                .or_default()
                .push(json!({"file":path,"kinds":kinds}));
        }
        if path.extension().is_some_and(|ext| {
            ext == "reds" || ext == "redscripts" || ext == "modded" || ext == "exe"
        }) {
            let mut hints: BTreeSet<usize> = BTreeSet::new();
            for found in matcher.stems.find_iter(&data) {
                let stem = &matcher.stem_names[found.pattern().as_usize()];
                if let Some(ids) = matcher.stem_targets.get(stem) {
                    hints.extend(ids.iter().copied());
                }
            }
            for id in hints {
                script_name_hints
                    .entry(id)
                    .or_default()
                    .push(json!({"file":path}));
            }
        }
    }
    let rows: Vec<_> = candidates.iter().enumerate().map(|(id, candidate)| {
        let blocked = hits.contains_key(&id);
        let ambiguous = script_name_hints.contains_key(&id);
        let valid = details.get(&id).is_some_and(|detail| candidate.ends_with(".app") || matches!(detail["puppet_type"].as_str(), Some("NPCPuppet" | "gamePuppet")));
        json!({"path":candidate,"status":if blocked {"referenced"} else if ambiguous {"script_name_requires_review"} else if !valid {"not_a_validated_character_resource"} else {"no_known_references"},"details":details.get(&id),"references":hits.get(&id),"script_name_hints":script_name_hints.get(&id)})
    }).collect();
    let output = json!({"input_candidates":candidates.len(),"archives":archives.len(),"scanned_resources":scanned_resources,"scanned_bytes":scanned_bytes,"native_tables":factory_tables,"loose_files":loose.len(),"excluded_media":skipped_media,"errors":errors,"factory_rows":factory_rows,"candidates":rows,"runtime_verified":false,"limits":["Non-structural external buffers such as pixels/audio/geometry are excluded", "Dynamically constructed names and unknown native code behavior cannot be proven absent", "All archive versions scanned conservatively; installed loose scripts may include other mods", "No console or mod.io runtime validation performed"]});
    fs::write(&args[4], serde_json::to_vec_pretty(&output)?)?;
    println!(
        "Wrote deep audit of {} candidates to {}",
        candidates.len(),
        args[4]
    );
    Ok(())
}
