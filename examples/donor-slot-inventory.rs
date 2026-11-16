//! Turn a deep reference audit and native serialization receipts into disjoint slot candidates.
use anyhow::{Context, Result, ensure};
use base64::Engine as _;
use ghostline_red::archive::{depot_path_hash, read_archive};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

fn load(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| path.display().to_string())
}

fn text(value: &Value) -> &str {
    value
        .as_str()
        .or_else(|| value["$value"].as_str())
        .unwrap_or_default()
}

fn collect_refs(value: &Value, targets: &HashMap<u64, usize>, out: &mut Vec<usize>) {
    match value {
        Value::String(value) => {
            let hash = value.parse().unwrap_or_else(|_| depot_path_hash(value));
            if let Some(id) = targets.get(&hash) {
                out.push(*id);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_refs(value, targets, out);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_refs(value, targets, out);
            }
        }
        _ => {}
    }
}

fn archive_paths(root: &Path, paths: &mut Vec<std::path::PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            archive_paths(&path, paths)?;
        } else if path.extension().is_some_and(|ext| ext == "archive") {
            paths.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "usage: donor-slot-inventory DEEP_REPORT DECODE_RECEIPT OUTPUT [GAME]"
    );
    let deep = load(Path::new(&args[1]))?;
    let receipt = load(Path::new(&args[2]))?;
    let mut candidates = deep["candidates"].as_array().context("candidates")?.clone();
    let targets: HashMap<_, _> = candidates
        .iter()
        .enumerate()
        .map(|(id, value)| (depot_path_hash(text(&value["path"])), id))
        .collect();
    let ids: HashMap<_, _> = candidates
        .iter()
        .enumerate()
        .map(|(id, value)| (text(&value["path"]).to_owned(), id))
        .collect();
    let mut failures = Vec::new();
    let mut tables = Vec::new();
    let mut typed_refs: BTreeMap<usize, Vec<Value>> = BTreeMap::new();
    let mut decoded = 0;
    for job in receipt.as_array().context("native receipt")? {
        let resource = text(&job["resource"]);
        if !job["error"].is_null() {
            failures.push(job.clone());
            continue;
        }
        let document = load(Path::new(text(&job["output"])))?;
        let root = &document["Data"]["RootChunk"];
        let own_id = ids.get(resource).copied();
        if text(&root["$type"]) == "C2dArray" {
            let empty = Vec::new();
            let rows = if let Some(rows) = root["data"].as_array() {
                rows
            } else {
                let headers = root["headers"]
                    .as_array()
                    .context("empty native table headers")?;
                ensure!(headers.len() < 64, "large undecoded table: {resource}");
                let mut expected = vec![headers.len() as u8];
                for header in headers {
                    let header = text(header);
                    ensure!(
                        header.is_ascii() && header.len() < 64,
                        "unsupported undecoded table header: {resource}"
                    );
                    expected.push(if header.is_empty() {
                        0
                    } else {
                        0x80 | header.len() as u8
                    });
                    expected.extend(header.as_bytes());
                }
                expected.push(0);
                let bytes =
                    base64::engine::general_purpose::STANDARD.decode(text(&root["$rawTail"]))?;
                ensure!(
                    bytes == expected,
                    "native table is not an independently verified empty table: {resource}"
                );
                &empty
            };
            tables.push(json!({"resource":resource,"rows":rows.len(),"headers":root["headers"]}));
            for (row, value) in rows.iter().enumerate() {
                let mut found = Vec::new();
                collect_refs(value, &targets, &mut found);
                found.sort_unstable();
                found.dedup();
                for id in found {
                    typed_refs
                        .entry(id)
                        .or_default()
                        .push(json!({"resource":resource,"row":row,"values":value}));
                }
            }
        } else {
            let mut found = Vec::new();
            collect_refs(root, &targets, &mut found);
            found.sort_unstable();
            found.dedup();
            for id in found {
                if Some(id) != own_id {
                    typed_refs
                        .entry(id)
                        .or_default()
                        .push(json!({"resource":resource,"kind":"decoded_document_reference"}));
                }
            }
        }
        if let Some(id) = own_id {
            decoded += 1;
            let chunks = root["compiledData"]["Data"]["Chunks"].as_array();
            let puppet = chunks
                .and_then(|chunks| chunks.first())
                .map(|chunk| text(&chunk["$type"]))
                .unwrap_or_default();
            let names: Vec<_> = root["appearances"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|entry| text(&entry.get("Data").unwrap_or(entry)["name"]).to_owned())
                .collect();
            let valid = if resource.ends_with(".app") {
                text(&root["$type"]) == "appearanceAppearanceResource" && !names.is_empty()
            } else {
                matches!(puppet, "NPCPuppet" | "gamePuppet")
            };
            candidates[id]["native_details"] = json!({"root_type":root["$type"],"puppet_type":puppet,"appearance_names":names,"compiled_chunks":chunks.map(Vec::len),"valid_character_resource":valid});
        }
    }
    for (id, candidate) in candidates.iter_mut().enumerate() {
        let path = text(&candidate["path"]);
        let runtime_family = path.contains("\\appearances\\corpse\\")
            || path.contains("\\entities\\corpse\\")
            || path.contains("\\entities\\player\\");
        let status = if typed_refs.contains_key(&id) || candidate["references"].is_array() {
            "referenced"
        } else if runtime_family {
            "dynamic_use_not_cleared"
        } else if candidate["script_name_hints"].is_array() {
            "script_name_requires_review"
        } else if candidate["native_details"]["valid_character_resource"] != true {
            "not_a_validated_character_resource"
        } else {
            "statically_unreferenced"
        };
        candidate["status"] = json!(status);
        if runtime_family {
            candidate["exclusion_reason"] = json!(
                "Corpse and player resource families are withheld because runtime name-based selection has not been cleared; absence of literal paths does not establish non-use"
            );
        }
        candidate["decoded_references"] = json!(typed_refs.get(&id));
        candidate.as_object_mut().unwrap().remove("details");
    }
    let eligible: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate["status"] == "statically_unreferenced")
        .collect();
    let mut base_roots = Vec::new();
    let mut ep1_roots = Vec::new();
    let mut base_apps = Vec::new();
    let mut ep1_apps = Vec::new();
    for candidate in &eligible {
        let path = text(&candidate["path"]);
        match (path.starts_with("ep1\\"), path.ends_with(".app")) {
            (false, false) => base_roots.push(*candidate),
            (true, false) => ep1_roots.push(*candidate),
            (false, true) => base_apps.push(*candidate),
            (true, true) => ep1_apps.push(*candidate),
        }
    }
    let root_priority = |value: &&Value| {
        let path = text(&value["path"]);
        (
            usize::from(!path.contains("to_be_removed")),
            path.to_owned(),
        )
    };
    base_roots.sort_by_key(root_priority);
    ep1_roots.sort_by_key(root_priority);
    base_apps.sort_by_key(|value| text(&value["path"]).to_owned());
    ep1_apps.sort_by_key(|value| text(&value["path"]).to_owned());
    let base_count = base_roots.len().min(base_apps.len());
    let mut slots = Vec::new();
    let mut pair = |root: &Value, appearance: &Value, expansion: bool| {
        slots.push(json!({"id":format!("GLCHAR{:04}",slots.len()+1),"requires_phantom_liberty":expansion,"root":root["path"],"appearance":appearance["path"],"preserve_root_appearance_names":root["native_details"]["appearance_names"],"original_puppet_type":root["native_details"]["puppet_type"],"owner":null,"status":"statically_audited_unallocated","runtime_verified":false}));
    };
    for i in 0..base_count {
        pair(base_roots[i], base_apps[i], false);
    }
    let remaining_roots: Vec<_> = base_roots
        .iter()
        .skip(base_count)
        .chain(ep1_roots.iter())
        .copied()
        .collect();
    let remaining_apps: Vec<_> = ep1_apps
        .iter()
        .chain(base_apps.iter().skip(base_count))
        .copied()
        .collect();
    let expansion_count = remaining_roots.len().min(remaining_apps.len());
    for i in 0..expansion_count {
        pair(remaining_roots[i], remaining_apps[i], true);
    }
    let mut statuses: BTreeMap<String, usize> = BTreeMap::new();
    for candidate in &candidates {
        *statuses
            .entry(text(&candidate["status"]).to_owned())
            .or_default() += 1;
    }
    let output = json!({"format":"ghostline-character-slot-inventory","schema_version":1,"audit_date":"2026-11-08","scope":"The 492 previously screened character entity/appearance resources; not a global maximum over all game paths","counts":{"input":candidates.len(),"decoded_candidates":decoded,"base_roots":base_roots.len(),"ep1_roots":ep1_roots.len(),"base_appearances":base_apps.len(),"ep1_appearances":ep1_apps.len(),"base_game_pairs":base_count,"additional_expansion_pairs":expansion_count,"total_pairs":slots.len(),"statuses":statuses},"scan":{"archives":deep["archives"],"resources":deep["scanned_resources"],"bytes":deep["scanned_bytes"],"loose_files":deep["loose_files"],"native_tables":tables.len(),"scan_errors":deep["errors"],"decode_failures":failures},"allocation_rules":["A root and appearance path may belong to only one installed mod; reserve the whole pair, not just an appearance name", "Preserve required native root appearance names; replace both files as a coherent NPC assembly", "Do not overwrite shared dependency meshes, rigs, textures or materials; separately reserve any additional replacement paths", "The inventory is not runtime-certified; validate a representative replacement before treating the pool as production-ready", "Coordinate allocations between quest series; arbitrary external mods may already replace any donor path"],"slots":slots,"candidates":candidates,"tables":tables});
    let mut output = output;
    let roots: std::collections::HashSet<_> = output["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| text(&slot["root"]))
        .collect();
    let appearances: std::collections::HashSet<_> = output["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| text(&slot["appearance"]))
        .collect();
    ensure!(
        roots.len() == output["slots"].as_array().unwrap().len()
            && appearances.len() == roots.len(),
        "slot replacement paths overlap"
    );
    if Path::new(&args[3]).exists() {
        let existing = load(Path::new(&args[3]))?;
        ensure!(
            existing["slots"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|slot| slot["owner"].is_null()),
            "refusing to overwrite an inventory containing allocated slots"
        );
    }
    if let Some(game) = args.get(4) {
        let game = Path::new(game);
        let mut vanilla = Vec::new();
        for folder in ["archive/pc/content", "archive/pc/ep1", "archive/pc/hot"] {
            let mut paths = Vec::new();
            archive_paths(&game.join(folder), &mut paths)?;
            paths.sort();
            vanilla.extend(paths);
        }
        let mut identities = HashMap::new();
        for archive in vanilla {
            for entry in read_archive(&archive)?.entries {
                if targets.contains_key(&entry.name_hash) {
                    identities.insert(
                        entry.name_hash,
                        json!({"archive":archive,"sha1":entry.sha1,"bytes":entry.size}),
                    );
                }
            }
        }
        let mut mods = Vec::new();
        archive_paths(&game.join("archive/pc/mod"), &mut mods)?;
        archive_paths(&game.join("mods"), &mut mods)?;
        let mut collisions: HashMap<u64, Vec<Value>> = HashMap::new();
        for archive in &mods {
            for entry in read_archive(archive)?.entries {
                if targets.contains_key(&entry.name_hash) {
                    collisions
                        .entry(entry.name_hash)
                        .or_default()
                        .push(json!({"archive":archive,"sha1":entry.sha1}));
                }
            }
        }
        let mut conflicting = 0;
        for slot in output["slots"].as_array_mut().unwrap() {
            let root = depot_path_hash(text(&slot["root"]));
            let appearance = depot_path_hash(text(&slot["appearance"]));
            slot["source_fingerprints"] =
                json!({"root":identities.get(&root),"appearance":identities.get(&appearance)});
            slot["installed_mod_overrides"] =
                json!({"root":collisions.get(&root),"appearance":collisions.get(&appearance)});
            if collisions.contains_key(&root) || collisions.contains_key(&appearance) {
                conflicting += 1;
            }
        }
        output["scan"]["installed_mod_archives"] = json!(mods.len());
        output["counts"]["pairs_overridden_in_current_install"] = json!(conflicting);
    }
    fs::write(&args[3], serde_json::to_vec_pretty(&output)?)?;
    println!("{}", serde_json::to_string_pretty(&output["counts"])?);
    Ok(())
}
