use aho_corasick::AhoCorasick;
use anyhow::{Context, Result, ensure};
use ghostline_red::{archive, codec, cr2w, redpackage, schema::RedSchema};
use rayon::prelude::*;
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

fn archive_paths(root: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            archive_paths(&path, paths)?;
        } else if path.extension().is_some_and(|e| e == "archive") {
            paths.push(path);
        }
    }
    Ok(())
}

pub(super) fn run(
    game: &Path,
    names_file: &Path,
    patterns_file: &Path,
    resources_file: Option<&Path>,
    all_resource_types: bool,
    include_event_buffers: bool,
    schema_file: Option<&Path>,
    output: &Path,
) -> Result<()> {
    let root = game.join("archive/pc");
    let mut paths = Vec::new();
    let schema = Some(RedSchema::load(Some(game), schema_file)?);
    let classes = schema.as_ref().map(RedSchema::class_names);
    for folder in ["content", "ep1"] {
        archive_paths(&root.join(folder), &mut paths)?;
    }
    paths.sort();
    let mut names = fs::read_to_string(names_file)?
        .lines()
        .map(|s| s.trim().replace('/', "\\"))
        .filter(|s| !s.is_empty())
        .map(|s| (archive::depot_path_hash(&s), s))
        .collect::<HashMap<_, _>>();
    let patterns = fs::read_to_string(patterns_file)?
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(!patterns.is_empty(), "no effect-name patterns supplied");
    let requested = resources_file
        .map(|path| -> Result<HashSet<u64>> {
            let mut hashes = HashSet::new();
            for line in fs::read_to_string(path)?.lines() {
                let name = line.trim().replace('/', "\\");
                if !name.is_empty() {
                    let hash = match name.parse::<u64>() {
                        Ok(hash) => hash,
                        Err(_) => {
                            let hash = archive::depot_path_hash(&name);
                            names.insert(hash, name);
                            hash
                        }
                    };
                    hashes.insert(hash);
                }
            }
            ensure!(!hashes.is_empty(), "resource list is empty");
            Ok(hashes)
        })
        .transpose()?;
    let mut needles = patterns
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect::<Vec<_>>();
    for pattern in &patterns {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for b in pattern.bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
        needles.push(hash.to_le_bytes().to_vec());
    }
    let matcher = AhoCorasick::new(&needles)?;
    let mut exact_patterns = HashMap::new();
    for (index, pattern) in patterns.iter().enumerate() {
        exact_patterns.insert(pattern.clone(), index);
        let hash_bytes: [u8; 8] = needles[index + patterns.len()].as_slice().try_into()?;
        exact_patterns.insert(u64::from_le_bytes(hash_bytes).to_string(), index);
    }
    let mut indices = Vec::new();
    let mut selected = HashMap::new();
    for path in &paths {
        let index = archive::read_archive(path)?;
        for (i, entry) in index.entries.iter().enumerate() {
            selected.insert(entry.name_hash, (indices.len(), i));
        }
        indices.push(index);
    }
    let extensions = [
        ".scene",
        ".questphase",
        ".anims",
        ".workspot",
        ".animgraph",
        ".actionanimdb",
    ];
    let missing = requested
        .as_ref()
        .map(|hashes| {
            hashes
                .iter()
                .filter(|hash| !selected.contains_key(hash))
                .map(|hash| names.get(hash).cloned().unwrap_or_else(|| hash.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let jobs = selected
        .into_iter()
        .filter(|(hash, _)| {
            if let Some(requested) = &requested {
                return requested.contains(hash);
            }
            if all_resource_types {
                return true;
            }
            names
                .get(hash)
                .is_none_or(|p| extensions.iter().any(|e| p.ends_with(e)))
        })
        .collect::<Vec<_>>();
    let done = AtomicUsize::new(0);
    let resource_count = AtomicUsize::new(0);
    let main_bytes = std::sync::atomic::AtomicU64::new(0);
    eprintln!(
        "Scanning {} known event resources and unnamed assets",
        jobs.len()
    );
    let rows=jobs.par_iter().filter_map(|(hash,(a,i))| {
        let count=done.fetch_add(1,Ordering::Relaxed)+1;
        if count%2000==0 {eprintln!("scanned {count}/{}",jobs.len());}
        let name=names.get(hash);
        let result=(||->Result<Option<serde_json::Value>> {
            // Audio/video payloads have no CR2W event tables. Inspect their raw
            // signature before allocating or decompressing a full main segment.
            let entry = &indices[*a].entries[*i];
            let segment = indices[*a].segments.get(entry.segments_start as usize)
                .context("missing main archive segment")?;
            let mut reader = fs::File::open(&paths[*a])?;
            reader.seek(SeekFrom::Start(segment.offset))?;
            let mut magic = [0_u8; 4];
            reader.read_exact(&mut magic)?;
            if &magic != b"CR2W" && &magic != b"KARK" { return Ok(None); }
            let bytes=archive::extract_entry_main_bytes(&paths[*a],&indices[*a],*i)?;
            main_bytes.fetch_add(bytes.len() as u64,Ordering::Relaxed);
            if !bytes.starts_with(b"CR2W") {return Ok(None)}
            let file=cr2w::inspect_bytes(&bytes).context("main metadata tables")?;
            let class=file.exports.first().map(|e|e.class_name.as_str()).unwrap_or("");
            let relevant=matches!(class,"scnSceneResource"|"questQuestPhaseResource"|"animAnimSet"|"workWorkspotResource"|"animAnimGraph"|"animActionAnimDatabase");
            if !all_resource_types&&name.is_none()&&!relevant {return Ok(None)}
            resource_count.fetch_add(1, Ordering::Relaxed);
            let mut hits=HashSet::new();
            for hit in matcher.find_overlapping_iter(&bytes) {hits.insert(hit.pattern().as_usize()%patterns.len());}
            let mut buffer_hits=HashSet::new();
            let mut package_uses = Vec::new();
            if include_event_buffers && matches!(class,
                "entEntityTemplate" | "appearanceAppearanceResource" | "animAnimGraph" |
                "workWorkspotResource" | "gameEffectSet" | "scnSceneResource" |
                "questQuestPhaseResource" | "worldEffect") && !file.buffers.is_empty() {
                let full = archive::extract_entry_bytes(&paths[*a], &indices[*a], *i)?;
                for buffer_index in 0..file.buffers.len() {
                    let decoded = codec::decode_table_buffer(&full, buffer_index)
                        .with_context(|| format!("metadata buffer {buffer_index}"))?;
                    for hit in matcher.find_overlapping_iter(&decoded) {
                        let pattern = hit.pattern().as_usize()%patterns.len();
                        hits.insert(pattern);
                        buffer_hits.insert(pattern);
                    }
                    if let (Some(schema), Some(classes)) = (&schema, &classes) {
                        if !matcher.is_match(&decoded) { continue; }
                        let sections = decoded.get(2..4).map(|b| u16::from_le_bytes([b[0], b[1]]));
                        if decoded.first().is_some_and(|v| (2..=4).contains(v)) && sections.is_some_and(|s| (6..=7).contains(&s)) {
                            let package = redpackage::decode(&decoded, classes, redpackage::PackageSettings {
                                imports_as_hash: redpackage::imports_are_hashed(&decoded)?,
                                handle_id_base: 0,
                                enums: Some(&schema.enums),
                                bitfields: Some(&schema.bitfields),
                            }).with_context(|| format!("RED package buffer {buffer_index}"))?;
                            collect_uses(&package, "$", &[], &exact_patterns, &patterns, &mut package_uses);
                        }
                    }
                }
            }
            if all_resource_types && hits.is_empty() { return Ok(None); }
            Ok(Some(json!({"depot":name,"hash":hash.to_string(),"archive":paths[*a],"class":class,"main_bytes":bytes.len(),"hits":hits.into_iter().map(|i|patterns[i].clone()).collect::<Vec<_>>(),"matching_imports":file.imports.iter().filter(|i|exact_patterns.contains_key(&i.depot_path)).collect::<Vec<_>>(),"buffer_hits":buffer_hits.into_iter().map(|i|patterns[i].clone()).collect::<Vec<_>>(),"package_uses":package_uses,"sha1":indices[*a].entries[*i].sha1})))
        })();
        match result {Ok(row)=>row,Err(e)=>Some(json!({"depot":name,"hash":hash.to_string(),"archive":paths[*a],"error":format!("{e:#}")}))}
    }).collect::<Vec<_>>();
    let failures = rows.iter().filter(|r| r.get("error").is_some()).count();
    let hit_files = rows
        .iter()
        .filter(|r| r["hits"].as_array().is_some_and(|a| !a.is_empty()))
        .count();
    fs::write(
        output,
        serde_json::to_vec_pretty(
            &json!({"patterns":patterns,"all_resource_types":all_resource_types,"include_event_buffers":include_event_buffers,"requested_resources":requested.as_ref().map(HashSet::len),"missing_resources":missing,"scanned_entries":jobs.len(),"main_bytes_read":main_bytes.load(Ordering::Relaxed),"event_resources":resource_count.load(Ordering::Relaxed),"failures":failures,"hit_resources":hit_files,"resources":rows}),
        )?,
    )?;
    eprintln!(
        "Done: {} event resources, {hit_files} hits, {failures} failures",
        resource_count.load(Ordering::Relaxed)
    );
    Ok(())
}

fn collect_uses(
    value: &serde_json::Value,
    path: &str,
    types: &[String],
    exact: &HashMap<String, usize>,
    patterns: &[String],
    uses: &mut Vec<serde_json::Value>,
) {
    match value {
        serde_json::Value::Object(object) => {
            let mut types = types.to_vec();
            if let Some(class) = object.get("$type").and_then(serde_json::Value::as_str) {
                types.push(class.to_owned());
                if class == "entEffectDesc" {
                    if let Some(name) = object
                        .get("effectName")
                        .and_then(|v| v.get("$value"))
                        .and_then(serde_json::Value::as_str)
                    {
                        if let Some(index) = exact.get(name) {
                            let automatic = object.get("isAutoSpawn").is_some_and(|v| {
                                v.as_bool()
                                    .unwrap_or_else(|| v.as_u64().is_some_and(|v| v != 0))
                            });
                            uses.push(json!({"pattern":patterns[*index],"path":path,"types":types,"descriptor":true,"auto_spawn":automatic}));
                        }
                    }
                }
            }
            for (key, child) in object {
                collect_uses(
                    child,
                    &format!("{path}.{key}"),
                    &types,
                    exact,
                    patterns,
                    uses,
                );
            }
        }
        serde_json::Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                collect_uses(
                    child,
                    &format!("{path}[{index}]"),
                    types,
                    exact,
                    patterns,
                    uses,
                );
            }
        }
        serde_json::Value::String(text) => {
            if let Some(index) = exact.get(text) {
                uses.push(json!({"pattern":patterns[*index],"path":path,"types":types,"descriptor":types.iter().any(|t| t == "entEffectDesc")}));
            }
        }
        _ => {}
    }
}
