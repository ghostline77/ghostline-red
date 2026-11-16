//! Native extraction of mesh materials and their render-time dependencies.

use crate::{
    archive::{self, ArchiveIndex},
    codec,
    schema::RedSchema,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::OsStr,
    fmt::Write as _,
    fs, io,
    path::{Component, Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, OnceLock},
};
use texture2ddecoder::{
    decode_bc1, decode_bc3, decode_bc4, decode_bc5, decode_bc6_signed, decode_bc6_unsigned,
    decode_bc7,
};
use thiserror::Error;

const DEPOT_EXTENSIONS: &[&str] = &[
    "mi",
    "w2mi",
    "mt",
    "remt",
    "xbm",
    "mlmask",
    "mlsetup",
    "mltemplate",
    "gradient",
    "hp",
    "texarray",
];

struct ExportedDependency {
    artifact: PathBuf,
    size: u64,
    modified: std::time::SystemTime,
    dependencies: Vec<String>,
}

#[derive(Default)]
struct DependencySlot {
    preparation: Mutex<()>,
    completed: OnceLock<ExportedDependency>,
}

static DEPENDENCY_LOCKS: LazyLock<Mutex<HashMap<PathBuf, Arc<DependencySlot>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Error)]
pub enum MaterialError {
    #[error("could not access material data: {0}")]
    Io(#[from] io::Error),
    #[error("could not read game archive: {0}")]
    Archive(#[from] archive::ArchiveError),
    #[error("could not decode material CR2W: {0}")]
    Codec(#[from] codec::CodecError),
    #[error("could not decode embedded bytes: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("could not encode material JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("malformed material data: {0}")]
    Malformed(String),
    #[error("unsupported texture compression: {0}")]
    UnsupportedTexture(String),
    #[error("texture decoder failed: {0}")]
    Texture(String),
    #[error("game resource was not found in the indexed archives: {0}")]
    ResourceNotFound(String),
    #[error("failed to export dependency {path}: {source}")]
    Dependency {
        path: String,
        #[source]
        source: Box<MaterialError>,
    },
}

#[derive(Debug)]
struct ArchiveSource {
    path: PathBuf,
    index: ArchiveIndex,
}

/// An in-memory hash index over the game's archive files.
#[derive(Debug)]
pub struct ArchiveSet {
    sources: Vec<ArchiveSource>,
    locations: HashMap<u64, (usize, usize)>,
    template_cache: Mutex<HashMap<String, Map<String, Value>>>,
    payload_cache: Mutex<PayloadCache>,
    payload_locks: [Mutex<()>; 64],
}

#[derive(Debug, Default)]
struct PayloadCache {
    entries: HashMap<u64, Arc<Vec<u8>>>,
    bytes: usize,
}

#[derive(Debug)]
struct MaterialRecord {
    name: String,
    base_material: String,
    material_template: String,
    enable_mask: bool,
    data: Map<String, Value>,
    resource_fallbacks: Vec<Value>,
}

#[derive(Debug, Default)]
pub struct ExportSummary {
    pub materials: usize,
    pub dependencies: usize,
    pub textures: usize,
    pub masks: usize,
}

impl ArchiveSet {
    /// Identifies the effective resource index, including stored payload hashes.
    ///
    /// The identity changes when a selected resource's archive metadata changes.
    /// Open a new archive set after installing a game patch.
    #[must_use]
    pub fn content_identity(&self) -> String {
        use sha1::{Digest as _, Sha1};
        let mut entries: Vec<_> = self.locations.iter().collect();
        entries.sort_unstable_by_key(|(hash, _)| **hash);
        let mut digest = Sha1::new();
        for (&hash, &(source, index)) in entries {
            let entry = &self.sources[source].index.entries[index];
            digest.update(hash.to_le_bytes());
            digest.update(entry.size.to_le_bytes());
            digest.update(entry.sha1.as_bytes());
        }
        format!("{:x}", digest.finalize())
    }
    /// Indexes every `.archive` beneath a game archive directory.
    ///
    /// # Errors
    ///
    /// Returns [`MaterialError`] when the directory cannot be traversed or an
    /// archive index is invalid.
    pub fn open(root: &Path) -> Result<Self, MaterialError> {
        let mut paths = Vec::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for item in fs::read_dir(&directory)? {
                let path = item?.path();
                if path.is_dir() {
                    pending.push(path);
                } else if path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("archive"))
                {
                    paths.push(path);
                }
            }
        }
        Self::from_paths(paths)
    }

    /// Indexes an explicit archive selection in sorted path order.
    ///
    /// Later paths win duplicate resource hashes. This is the native reader's
    /// deterministic ordering, not a model of runtime mod substitutions.
    ///
    /// # Errors
    ///
    /// Returns [`MaterialError`] if an archive cannot be read or indexed.
    pub fn from_paths(mut paths: Vec<PathBuf>) -> Result<Self, MaterialError> {
        paths.sort();
        paths.dedup();
        let indexed: Result<Vec<_>, archive::ArchiveError> = paths
            .par_iter()
            .map(|path| {
                Ok(ArchiveSource {
                    path: path.clone(),
                    index: archive::read_archive(path)?,
                })
            })
            .collect();
        let sources = indexed?;
        let mut locations = HashMap::new();
        for (source_index, source) in sources.iter().enumerate() {
            for (entry_index, entry) in source.index.entries.iter().enumerate() {
                // Later archives take precedence, matching the game's patch-like
                // replacement behavior while remaining deterministic.
                locations.insert(entry.name_hash, (source_index, entry_index));
            }
        }
        Ok(Self {
            sources,
            locations,
            template_cache: Mutex::new(HashMap::new()),
            payload_cache: Mutex::new(PayloadCache::default()),
            payload_locks: std::array::from_fn(|_| Mutex::new(())),
        })
    }

    /// Extracts an exact depot path using the already-built archive index.
    ///
    /// # Errors
    ///
    /// Returns [`MaterialError`] when the resource is absent or its archive
    /// payload cannot be read or decompressed.
    pub fn read_resource(&self, depot_path: &str) -> Result<Vec<u8>, MaterialError> {
        let hash = resource_hash(depot_path);
        self.read_resource_hash(hash).map_err(|error| match error {
            MaterialError::ResourceNotFound(_) => {
                MaterialError::ResourceNotFound(format!("{depot_path} ({hash})"))
            }
            other => other,
        })
    }

    /// Checks native archive membership without decompressing the resource.
    pub fn contains_resource_hash(&self, hash: u64) -> bool {
        self.locations.contains_key(&hash)
    }

    /// Extracts a resource identified by its native depot hash.
    ///
    /// # Errors
    /// Returns [`MaterialError`] for an absent resource or invalid archive payload.
    pub fn read_resource_hash(&self, hash: u64) -> Result<Vec<u8>, MaterialError> {
        const BUDGET: usize = 256 * 1024 * 1024;
        let _guard = self.payload_locks[(hash & 63) as usize]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(bytes) = self
            .payload_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .get(&hash)
            .cloned()
        {
            return Ok((*bytes).clone());
        }
        let &(source_index, entry_index) = self
            .locations
            .get(&hash)
            .ok_or_else(|| MaterialError::ResourceNotFound(format!("{hash}")))?;
        let source = &self.sources[source_index];
        let bytes = archive::extract_entry_bytes(&source.path, &source.index, entry_index)?;
        if bytes.len() <= BUDGET / 4 {
            let mut cache = self
                .payload_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while cache.bytes + bytes.len() > BUDGET {
                let Some(old) = cache.entries.keys().next().copied() else {
                    break;
                };
                if let Some(removed) = cache.entries.remove(&old) {
                    cache.bytes -= removed.len();
                }
            }
            cache.bytes += bytes.len();
            cache.entries.insert(hash, Arc::new(bytes.clone()));
        }
        Ok(bytes)
    }

    fn template_values(
        &self,
        depot: &str,
        schema: &RedSchema,
    ) -> Result<Map<String, Value>, MaterialError> {
        if let Some(values) = self
            .template_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(depot)
            .cloned()
        {
            return Ok(values);
        }
        let bytes = match self.read_resource(depot) {
            Ok(bytes) => bytes,
            Err(MaterialError::ResourceNotFound(_)) => {
                // Some shipped instances retain an uncooked template path
                // absent from archives. Preserve their explicit parameters;
                // PBR still rejects any required resource it cannot resolve.
                self.template_cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(depot.to_owned(), Map::new());
                return Ok(Map::new());
            }
            Err(error) => return Err(error),
        };
        let document = decode_bytes(&bytes, schema)?;
        let root = path(&document, &["Data", "RootChunk"])?;
        let mut values = Map::new();
        if let Some(parameters) = root.get("parameters") {
            collect_template_values(parameters, &mut values);
        }
        self.template_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(depot.to_owned(), values.clone());
        Ok(values)
    }
}

fn resource_hash(path: &str) -> u64 {
    path.strip_prefix("__hash__/")
        .or_else(|| path.strip_prefix(r"__hash__\"))
        .and_then(|value| value.split('.').next())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| archive::depot_path_hash(path))
}

fn collect_template_values(value: &Value, values: &mut Map<String, Value>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_template_values(item, values);
            }
        }
        Value::Object(object) => {
            if let Some(name) = object.get("parameterName").and_then(|v| red_string(v).ok()) {
                let kind = object
                    .get("$type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let field = match kind {
                    "CMaterialParameterScalar" => Some("scalar"),
                    "CMaterialParameterVector" => Some("vector"),
                    "CMaterialParameterColor" => Some("color"),
                    "CMaterialParameterTexture"
                    | "CMaterialParameterCube"
                    | "CMaterialParameterTextureArray" => Some("texture"),
                    "CMaterialParameterMultilayerMask" => Some("mask"),
                    "CMaterialParameterMultilayerSetup" => Some("setup"),
                    "CMaterialParameterGradient" => Some("gradient"),
                    _ => None,
                };
                if let Some(field) = field {
                    if let Some(v) = object.get(field) {
                        values.insert(name, normalize_value(v));
                    } else if field == "scalar" {
                        values.insert(name, json!(0.0));
                    } else if field == "vector" {
                        values.insert(name, json!({"X":0.0,"Y":0.0,"Z":0.0,"W":0.0}));
                    } else if field == "color" {
                        values.insert(name, json!({"Red":0,"Green":0,"Blue":0,"Alpha":0}));
                    }
                }
            } else {
                for item in object.values() {
                    collect_template_values(item, values);
                }
            }
        }
        _ => {}
    }
}

/// Writes a WolvenKit-compatible `.Material.json` and native material repo.
///
/// # Errors
///
/// Returns [`MaterialError`] for malformed mesh/material resources, missing
/// dependencies, unsupported textures, archive failures, or output I/O errors.
pub fn export_mesh_materials(
    mesh_input: &Path,
    schema: &RedSchema,
    archives: &ArchiveSet,
    repository: &Path,
    sidecar: &Path,
    appearance: Option<&str>,
    defer_dependency_errors: bool,
) -> Result<ExportSummary, MaterialError> {
    let mesh = codec::decode_wkit_with_red_schema(mesh_input, schema)?;
    export_mesh_materials_from_document(
        &mesh,
        schema,
        archives,
        repository,
        sidecar,
        appearance,
        None,
        defer_dependency_errors,
    )
}

/// Resolves and exports materials from an already decoded mesh document.
///
/// Reuses native mesh decoding when geometry and materials are prepared together.
/// When supplied, `chunk_indices` selects native render-chunk indices within the
/// appearance. Materials used only by other chunks are not resolved or exported.
///
/// # Errors
///
/// Returns [`MaterialError`] for malformed data, absent appearances/dependencies,
/// unsupported texture formats or output errors.
#[expect(
    clippy::too_many_lines,
    reason = "the function keeps the sidecar's mesh-to-material mapping in one auditable flow"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "chunk selection shares the existing export inputs"
)]
pub fn export_mesh_materials_from_document(
    mesh: &Value,
    schema: &RedSchema,
    archives: &ArchiveSet,
    repository: &Path,
    sidecar: &Path,
    appearance: Option<&str>,
    chunk_indices: Option<&[usize]>,
    defer_dependency_errors: bool,
) -> Result<ExportSummary, MaterialError> {
    let embedded_resources = embedded_resource_documents(mesh)?;
    let root = path(mesh, &["Data", "RootChunk"])?;
    let entries = array(root, "materialEntries")?;
    let local_buffer = root.get("localMaterialBuffer");
    let embedded_bytes = local_buffer
        .and_then(|buffer| buffer.get("rawData"))
        .and_then(|raw_data| raw_data.get("Bytes"))
        .and_then(Value::as_str)
        .map(|bytes| STANDARD.decode(bytes))
        .transpose()?
        .unwrap_or_default();
    let headers = local_buffer
        .and_then(|buffer| buffer.get("rawDataHeaders"))
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let external = root
        .get("externalMaterials")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let preloaded_external = root
        .get("preloadExternalMaterials")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let all_appearances = decode_appearances(root)?;
    let selected_names = appearance
        .map(|name| {
            all_appearances
                .get(name)
                .and_then(Value::as_array)
                .ok_or_else(|| malformed("requested mesh appearance was not found"))
                .and_then(|values| {
                    if let Some(indices) = chunk_indices {
                        if values.is_empty() && !indices.is_empty() {
                            return Err(malformed("selected appearance has no chunk materials"));
                        }
                        indices
                            .iter()
                            .map(|index| red_string(&values[index % values.len()]))
                            .collect::<Result<HashSet<_>, _>>()
                    } else {
                        values
                            .iter()
                            .map(red_string)
                            .collect::<Result<HashSet<_>, _>>()
                    }
                })
        })
        .transpose()?;

    let mut materials = Vec::with_capacity(entries.len());
    let mut material_embedded = Vec::with_capacity(entries.len());
    for entry in entries {
        // Omitted CName properties carry the native None default. These
        // entries may be unused, or explicitly bound by a None chunk name.
        let name = entry
            .get("name")
            .map(red_string)
            .transpose()?
            .unwrap_or_else(|| "None".to_owned());
        if selected_names
            .as_ref()
            .is_some_and(|selected| !selected.contains(&name))
        {
            continue;
        }
        let index = usize_value(entry.get("index").unwrap_or(&Value::Null)).unwrap_or(0);
        let mut source_correction = None;
        let document = if entry
            .get("isLocalInstance")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            if let Some(header) = headers.get(index) {
                let offset = usize_value(header.get("offset").unwrap_or(&Value::Null)).unwrap_or(0);
                let size = usize_value(
                    header
                        .get("size")
                        .ok_or_else(|| malformed("local material header has no size"))?,
                )
                .ok_or_else(|| malformed("local material size is invalid"))?;
                let end = offset
                    .checked_add(size)
                    .ok_or_else(|| malformed("local material byte range overflows"))?;
                decode_bytes(
                    embedded_bytes.get(offset..end).ok_or_else(|| {
                        malformed("local material byte range is outside the buffer")
                    })?,
                    schema,
                )?
            } else {
                let preloaded = root
                    .get("preloadLocalMaterialInstances")
                    .and_then(Value::as_array)
                    .and_then(|materials| materials.get(index))
                    .and_then(|material| material.get("Data"))
                    .ok_or_else(|| {
                        malformed(
                            "local material index is outside both the buffer and preload table",
                        )
                    })?;
                json!({"Data": {"RootChunk": preloaded}})
            }
        } else {
            let depot_path = external
                .get(index)
                .or_else(|| preloaded_external.get(index))
                .ok_or_else(|| malformed("external material index is outside both material tables"))
                .and_then(red_depot_path)?;
            if matches!(extension(&depot_path), Some("mt" | "remt")) {
                // Some native material entries reference a shader directly,
                // using its declared parameter defaults without an instance.
                json!({"Data":{"RootChunk":{"$type":"CMaterialInstance","baseMaterial":{"DepotPath":{"$value":depot_path}}}}})
            } else {
                let installed = installed_material_path(archives, &depot_path);
                if installed != depot_path {
                    source_correction = Some(json!({"missing":depot_path,"replacement":installed,
                        "reason":"same native material at its installed depot location"}));
                }
                decode_bytes(&archives.read_resource(installed)?, schema)?
            }
        };
        let mut embedded = embedded_resources.clone();
        let mut material = resolve_material(name, &document, archives, schema, 0, &mut embedded)?;
        material.resource_fallbacks.extend(source_correction);
        materials.push(material);
        material_embedded.push(embedded);
    }

    let appearances = if let Some(name) = appearance {
        Map::from_iter([(
            name.to_owned(),
            all_appearances
                .get(name)
                .cloned()
                .ok_or_else(|| malformed("requested mesh appearance was not found"))?,
        )])
    } else {
        all_appearances
    };
    let mut material_repositories = Vec::with_capacity(materials.len());
    let mut textures = HashSet::new();
    let mut dependencies = 0;
    let mut masks = 0;
    let mut dependency_errors = Vec::new();
    for (material, embedded) in materials.iter().zip(&material_embedded) {
        let material_repository = if embedded.is_empty() {
            repository.to_path_buf()
        } else {
            use sha1::Digest as _;
            let ordered: std::collections::BTreeMap<_, _> = embedded.iter().collect();
            let digest = sha1::Sha1::digest(serde_json::to_vec(&ordered)?);
            repository.join(format!("embedded-{digest:x}"))
        };
        fs::create_dir_all(&material_repository)?;
        let mut queue = VecDeque::new();
        collect_depot_paths(&Value::Object(material.data.clone()), &mut queue);
        let mut exporter =
            DependencyExporter::new(archives, schema, &material_repository, embedded);
        exporter.defer_errors = defer_dependency_errors;
        exporter.export(queue)?;
        dependencies += exporter.visited.len();
        masks += exporter.masks;
        textures.extend(exporter.textures);
        dependency_errors.extend(exporter.errors);
        material_repositories.push(material_repository);
    }

    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(repository)?;
    let texture_list: Vec<_> = textures.iter().cloned().collect();
    let material_values: Vec<_> = materials
        .iter()
        .zip(&material_repositories)
        .map(|(material,material_repository)| {
            json!({
                "Name": material.name,
                "BaseMaterial": material.base_material,
                "MaterialTemplate": material.material_template,
                "EnableMask": material.enable_mask,
                "Data": material.data,
                "ResourceFallbacks": material.resource_fallbacks,
                "MaterialRepo": material_repository.canonicalize().unwrap_or_else(|_|material_repository.clone()),
            })
        })
        .collect();
    let templates: Vec<_> = materials
        .iter()
        .map(|material| {
            json!({
                "Name": material.material_template,
                "EnableMask": material.enable_mask,
                "Data": template_defaults(&material.material_template),
            })
        })
        .collect();
    let document = json!({
        "Header": {"MaterialJsonVersion": "1.0.0"},
        "MaterialRepo": repository.canonicalize().unwrap_or_else(|_| repository.to_path_buf()),
        "Materials": material_values,
        "TexturesList": texture_list,
        "MaterialTemplates": templates,
        "Appearances": appearances,
        "DependencyErrors": dependency_errors,
    });
    fs::write(sidecar, serde_json::to_vec_pretty(&document)?)?;
    Ok(ExportSummary {
        materials: materials.len(),
        dependencies,
        textures: textures.len(),
        masks,
    })
}

fn embedded_resource_documents(mesh: &Value) -> Result<HashMap<String, Value>, MaterialError> {
    let mut resources = HashMap::new();
    let Some(files) = mesh
        .get("Data")
        .and_then(|data| data.get("EmbeddedFiles"))
        .and_then(Value::as_array)
    else {
        return Ok(resources);
    };
    for file in files {
        let depot_path = file
            .get("FileName")
            .ok_or_else(|| malformed("embedded resource has no file name"))
            .and_then(red_depot_path)?
            .replace('/', "\\")
            .to_lowercase();
        let content = file
            .get("Content")
            .ok_or_else(|| malformed("embedded resource has no content"))?;
        resources.insert(depot_path, json!({"Data": {"RootChunk": content}}));
    }
    Ok(resources)
}

/// Replaces typed parameters in a mesh-local material, retaining shader and textures.
/// The new material instance is serialized and decoded before updating its slot.
///
/// # Errors
/// Returns errors for absent slots, invalid buffers or failed serialization.
pub fn override_material_parameters(
    mesh: &mut Value,
    name: &str,
    parameters: &[Value],
    schema: &RedSchema,
) -> Result<(), MaterialError> {
    let root = &mut mesh["Data"]["RootChunk"];
    let entry = root["materialEntries"]
        .as_array()
        .and_then(|entries| entries.iter().find(|e| e["name"]["$value"] == name))
        .cloned()
        .ok_or_else(|| malformed(format!("material {name} absent")))?;
    let index = usize_value(&entry["index"]).unwrap_or(0);
    let mut bytes = root
        .pointer("/localMaterialBuffer/rawData/Bytes")
        .and_then(Value::as_str)
        .map(|v| STANDARD.decode(v))
        .transpose()?
        .unwrap_or_default();
    let headers = root
        .pointer("/localMaterialBuffer/rawDataHeaders")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut document = if entry["isLocalInstance"].as_bool() == Some(true) {
        if let Some(header) = headers.get(index) {
            let offset = usize_value(&header["offset"]).unwrap_or(0);
            let size =
                usize_value(&header["size"]).ok_or_else(|| malformed("material size absent"))?;
            decode_bytes(
                bytes
                    .get(offset..offset + size)
                    .ok_or_else(|| malformed("material buffer range"))?,
                schema,
            )?
        } else {
            json!({"Data":{"RootChunk":root["preloadLocalMaterialInstances"][index]["Data"]}})
        }
    } else {
        let base = root["externalMaterials"]
            .get(index)
            .or_else(|| root["preloadExternalMaterials"].get(index))
            .cloned()
            .ok_or_else(|| malformed("external material absent"))?;
        json!({"Data":{"RootChunk":{"$type":"CMaterialInstance","baseMaterial":base,"enableMask":false,"values":[]}}})
    };
    let values = document["Data"]["RootChunk"]["values"]
        .as_array_mut()
        .ok_or_else(|| malformed("material values absent"))?;
    for parameter in parameters {
        let key = parameter
            .as_object()
            .and_then(|v| v.keys().find(|k| !k.starts_with('$')))
            .ok_or_else(|| malformed("parameter name absent"))?;
        values.retain(|v| v.get(key).is_none());
        values.push(parameter.clone());
    }
    let staging = tempfile::tempdir()?;
    let raw = staging.path().join("material.json");
    let binary = staging.path().join("material.mi");
    fs::write(&raw, serde_json::to_vec(&document)?)?;
    crate::writer::write(&raw, &binary, schema)
        .map_err(|e| malformed(format!("material serialization: {e}")))?;
    let cooked = fs::read(binary)?;
    let readback = decode_bytes(&cooked, schema)?;
    for parameter in parameters {
        let key = parameter
            .as_object()
            .and_then(|v| v.keys().find(|k| !k.starts_with('$')))
            .ok_or_else(|| malformed("parameter name absent"))?;
        if !readback["Data"]["RootChunk"]["values"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|p| p[key] == parameter[key])
        {
            return Err(malformed(format!(
                "material parameter {key} changed in serialization"
            )));
        }
    }
    let offset = bytes.len();
    bytes.extend(&cooked);
    root["localMaterialBuffer"]["rawData"]["Bytes"] = Value::String(STANDARD.encode(bytes));
    let mut headers = headers;
    let new_index = headers.len();
    headers.push(json!({"$type":"meshLocalMaterialHeader","offset":offset,"size":cooked.len()}));
    root["localMaterialBuffer"]["rawDataHeaders"] = Value::Array(headers);
    let entry = root["materialEntries"]
        .as_array_mut()
        .and_then(|entries| entries.iter_mut().find(|e| e["name"]["$value"] == name))
        .ok_or_else(|| malformed("material entry disappeared"))?;
    entry["isLocalInstance"] = json!(true);
    entry["index"] = json!(new_index);
    Ok(())
}

/// Decodes mesh-local material instances, including opaque CR2W buffer payloads.
///
/// # Errors
/// Returns an error for invalid buffer ranges or undecodable material instances.
pub fn local_material_documents(
    mesh: &Value,
    schema: &RedSchema,
) -> Result<Vec<Value>, MaterialError> {
    let root = &mesh["Data"]["RootChunk"];
    let mut documents = Vec::new();
    if let Some(encoded) = root
        .pointer("/localMaterialBuffer/rawData/Bytes")
        .and_then(Value::as_str)
    {
        let bytes = STANDARD.decode(encoded)?;
        for header in root
            .pointer("/localMaterialBuffer/rawDataHeaders")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let offset = usize_value(&header["offset"]).unwrap_or(0);
            let size = usize_value(&header["size"])
                .ok_or_else(|| malformed("local material size absent"))?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| malformed("local material range overflow"))?;
            documents.push(decode_bytes(
                bytes
                    .get(offset..end)
                    .ok_or_else(|| malformed("local material buffer range"))?,
                schema,
            )?);
        }
    }
    for value in root["preloadLocalMaterialInstances"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if let Some(data) = value.get("Data") {
            documents.push(json!({"Data":{"RootChunk":data}}));
        }
    }
    Ok(documents)
}

/// Decodes the local material instances used by one named mesh appearance.
/// Unselected native material slots are not runtime dependencies of that look.
///
/// # Errors
/// Returns an error for missing appearances, invalid slots or malformed materials.
pub fn active_local_material_documents(
    mesh: &Value,
    appearance: &str,
    schema: &RedSchema,
) -> Result<Vec<Value>, MaterialError> {
    let root = &mesh["Data"]["RootChunk"];
    let selected = root["appearances"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|v| v.get("Data").unwrap_or(v))
        .find(|a| {
            a["name"]
                .get("$value")
                .unwrap_or(&a["name"])
                .as_str()
                .unwrap_or("default")
                == appearance
        })
        .ok_or_else(|| malformed(format!("mesh appearance {appearance} missing")))?;
    let names: std::collections::BTreeSet<_> = selected["chunkMaterials"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().or_else(|| v["$value"].as_str()))
        .collect();
    let indices: std::collections::BTreeSet<_> = root["materialEntries"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry["isLocalInstance"].as_bool() == Some(true)
                && names.contains(
                    entry["name"]["$value"]
                        .as_str()
                        .or_else(|| entry["name"].as_str())
                        .unwrap_or("None"),
                )
        })
        .map(|entry| usize_value(&entry["index"]).unwrap_or(0))
        .collect();
    let all = local_material_documents(mesh, schema)?;
    indices
        .into_iter()
        .map(|index| {
            all.get(index)
                .cloned()
                .ok_or_else(|| malformed(format!("active local material {index} absent")))
        })
        .collect()
}

fn installed_material_path<'a>(archives: &ArchiveSet, requested: &'a str) -> &'a str {
    if archives.contains_resource_hash(resource_hash(requested)) {
        return requested;
    }
    let replacement = match requested.replace('/', "\\").to_lowercase().as_str() {
        r"base\environment\decoration\garbage\trash\street_trash\street_trash.mi" => {
            r"base\environment\decoration\garbage\trash\textures\street_trash.mi"
        }
        _ => return requested,
    };
    if archives.contains_resource_hash(resource_hash(replacement)) {
        replacement
    } else {
        requested
    }
}

fn resolve_material(
    name: String,
    document: &Value,
    archives: &ArchiveSet,
    schema: &RedSchema,
    depth: usize,
    embedded_resources: &mut HashMap<String, Value>,
) -> Result<MaterialRecord, MaterialError> {
    if depth > 16 {
        return Err(malformed(
            "material inheritance is deeper than 16 resources",
        ));
    }
    for (name, document) in embedded_resource_documents(document)? {
        embedded_resources.entry(name).or_insert(document);
    }
    let root = path(document, &["Data", "RootChunk"])?;
    let base_material = root
        .get("baseMaterial")
        .map(red_depot_path)
        .transpose()?
        .unwrap_or_default();
    let mut data = Map::new();
    let mut resource_fallbacks = Vec::new();
    let mut material_template = base_material.clone();
    let mut enable_mask = root
        .get("enableMask")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if matches!(extension(&base_material), Some("mi" | "w2mi")) {
        let key = base_material.replace('/', "\\").to_lowercase();
        let parent_document = if let Some(document) = embedded_resources.get(&key) {
            document.clone()
        } else {
            decode_bytes(
                &archives.read_resource(installed_material_path(archives, &base_material))?,
                schema,
            )?
        };
        let parent = resolve_material(
            name.clone(),
            &parent_document,
            archives,
            schema,
            depth + 1,
            embedded_resources,
        )?;
        data = parent.data;
        resource_fallbacks = parent.resource_fallbacks;
        let installed = installed_material_path(archives, &base_material);
        if installed != base_material {
            resource_fallbacks.push(json!({"missing":base_material,"replacement":installed,
                "reason":"same native material at its installed depot location"}));
        }
        material_template = parent.material_template;
        enable_mask |= parent.enable_mask;
    }
    let defaults = if matches!(extension(&material_template), Some("mt" | "remt")) {
        archives.template_values(&material_template, schema)?
    } else {
        Map::new()
    };
    {
        for (key, value) in &defaults {
            data.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    if let Value::Object(defaults) = template_defaults(&material_template) {
        for (key, value) in defaults {
            data.entry(key).or_insert(value);
        }
    }
    if let Some(values) = root.get("values").and_then(Value::as_array) {
        for value in values {
            let object = value
                .as_object()
                .ok_or_else(|| malformed("material value is not an object"))?;
            for (key, item) in object {
                if !key.starts_with('$') {
                    let value = normalize_value(item);
                    let replacement = value
                        .as_str()
                        .filter(|p| {
                            matches!(extension(p), Some("xbm" | "mlmask" | "mlsetup"))
                                && !archives.locations.contains_key(&resource_hash(p))
                                && !embedded_resources
                                    .contains_key(&p.replace('/', "\\").to_lowercase())
                        })
                        .and_then(|missing| {
                            defaults
                                .get(key)
                                .filter(|fallback| {
                                    fallback.as_str().is_some_and(|p| {
                                        archives.locations.contains_key(&resource_hash(p))
                                    })
                                })
                                .map(|fallback| (missing, fallback))
                        });
                    if let Some((missing, fallback)) = replacement {
                        resource_fallbacks.push(json!({"parameter":key,"missing":missing,"replacement":fallback,"reason":"unresolved resource; authoritative shader-template default"}));
                        data.insert(key.clone(), fallback.clone());
                    } else {
                        data.insert(key.clone(), value);
                    }
                }
            }
        }
    }
    // Legacy metal instances sometimes retain maps removed from a surface
    // family. Its installed layer template is the native authority for that
    // exact family's channel inputs (for example plastic metalness is black).
    for (parameter, suffix, field) in [
        ("Metalness", "_m.xbm", "metalnessTexture"),
        ("Roughness", "_r.xbm", "roughnessTexture"),
        ("Normal", "_n.xbm", "normalTexture"),
        ("BaseColor", "_d.xbm", "colorTexture"),
    ] {
        let Some(missing) = data
            .get(parameter)
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        if archives.contains_resource_hash(resource_hash(&missing)) {
            continue;
        }
        let Some(stem) = missing.strip_suffix(suffix) else {
            continue;
        };
        let family = format!("{stem}.mltemplate");
        let Ok(bytes) = archives.read_resource(&family) else {
            continue;
        };
        let native = decode_bytes(&bytes, schema)?;
        let Some(replacement) = native.pointer(&format!("/Data/RootChunk/{field}")) else {
            continue;
        };
        let replacement = red_depot_path(replacement)?;
        if archives.contains_resource_hash(resource_hash(&replacement)) {
            data.insert(parameter.into(), json!(replacement));
            resource_fallbacks.push(json!({"parameter":parameter,"missing":missing,"replacement":replacement,
                "reason":"channel input declared by the installed native template of the exact same surface family","source":family}));
        }
    }
    // Shipped legacy shaders retain these old depot directories. Their exact
    // named textures are installed at the current native locations below.
    for (missing, replacement) in [
        (
            r"test\level_design\ld_kit\textures\ruler_checker_1m_wall.xbm",
            r"base\environment\ld_kit\ruler_checker_1m_wall.xbm",
        ),
        (
            r"base\environment\decoration\stickers\textures\decal_wall_stickers_d.xbm",
            r"base\environment\decoration\decals\stickers\textures\decal_wall_stickers_d.xbm",
        ),
    ] {
        if archives.contains_resource_hash(resource_hash(missing))
            || !archives.contains_resource_hash(resource_hash(replacement))
        {
            continue;
        }
        for (parameter, value) in &mut data {
            if value
                .as_str()
                .is_some_and(|path| path.replace('/', "\\").eq_ignore_ascii_case(missing))
            {
                *value = json!(replacement);
                resource_fallbacks.push(
                    json!({"parameter":parameter,"missing":missing,"replacement":replacement,
                    "reason":"same named texture at its installed native depot location"}),
                );
            }
        }
    }
    Ok(MaterialRecord {
        name,
        base_material,
        material_template,
        enable_mask,
        data,
        resource_fallbacks,
    })
}

fn decode_appearances(root: &Value) -> Result<Map<String, Value>, MaterialError> {
    let mut result = Map::new();
    for appearance in array(root, "appearances")? {
        let data = appearance.get("Data").unwrap_or(appearance);
        let name = data
            .get("name")
            .map(red_string)
            .transpose()?
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "default".to_owned());
        let materials = data
            .get("chunkMaterials")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("mesh appearance has no chunk material array"))?
            .iter()
            .map(red_string)
            .collect::<Result<Vec<_>, _>>()?;
        result.insert(name, json!(materials));
    }
    Ok(result)
}

struct DependencyExporter<'a> {
    archives: &'a ArchiveSet,
    schema: &'a RedSchema,
    repository: &'a Path,
    embedded_resources: &'a HashMap<String, Value>,
    visited: HashSet<String>,
    textures: HashSet<String>,
    masks: usize,
    defer_errors: bool,
    errors: Vec<String>,
}

impl<'a> DependencyExporter<'a> {
    fn new(
        archives: &'a ArchiveSet,
        schema: &'a RedSchema,
        repository: &'a Path,
        embedded_resources: &'a HashMap<String, Value>,
    ) -> Self {
        Self {
            archives,
            schema,
            repository,
            embedded_resources,
            visited: HashSet::new(),
            textures: HashSet::new(),
            masks: 0,
            defer_errors: false,
            errors: Vec::new(),
        }
    }

    fn export(&mut self, mut queue: VecDeque<String>) -> Result<(), MaterialError> {
        while let Some(depot_path) = queue.pop_front() {
            let normalized = depot_path.replace('/', "\\").to_lowercase();
            if !self.visited.insert(normalized.clone()) {
                continue;
            }
            let Some(extension) = extension(&normalized) else {
                continue;
            };
            if !DEPOT_EXTENSIONS.contains(&extension) {
                continue;
            }
            let result = (|| {
                let output = repo_path(self.repository, &normalized)?;
                let resource_lock = dependency_lock(&output);
                if let Some(cached) = resource_lock.completed.get()
                    && fs::metadata(&cached.artifact).is_ok_and(|metadata| {
                        metadata.is_file()
                            && metadata.len() == cached.size
                            && metadata.modified().ok() == Some(cached.modified)
                    })
                {
                    queue.extend(cached.dependencies.iter().cloned());
                    if extension == "xbm" {
                        self.textures.insert(normalized.clone());
                    } else if extension == "mlmask" {
                        self.masks += 1;
                    }
                    return Ok(());
                }
                let _resource_guard = resource_lock
                    .preparation
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if extension == "xbm" && output.with_extension("png").is_file() {
                    self.textures.insert(normalized.clone());
                    remember_dependency(&resource_lock, output.with_extension("png"), Vec::new())?;
                    return Ok(());
                }
                if extension == "mlmask" && output.with_extension("masklist").is_file() {
                    self.masks += 1;
                    remember_dependency(
                        &resource_lock,
                        output.with_extension("masklist"),
                        Vec::new(),
                    )?;
                    return Ok(());
                }
                let json_output = repo_path(self.repository, &format!("{normalized}.json"))?;
                if !matches!(extension, "xbm" | "mlmask") && json_output.is_file() {
                    let document: Value = serde_json::from_slice(&fs::read(&json_output)?)?;
                    let mut dependencies = VecDeque::new();
                    collect_depot_paths(&document, &mut dependencies);
                    remember_dependency(
                        &resource_lock,
                        json_output,
                        dependencies.iter().cloned().collect(),
                    )?;
                    queue.extend(dependencies);
                    return Ok(());
                }
                let mut document = if let Some(embedded) = self.embedded_resources.get(&normalized)
                {
                    embedded.clone()
                } else {
                    let bytes = self.archives.read_resource(&normalized)?;
                    decode_bytes(&bytes, self.schema)?
                };
                complete_blender_defaults(&mut document, extension)?;
                if extension == "mltemplate"
                    && self.embedded_resources.contains_key(&normalized)
                    && let Ok(bytes) = self.archives.read_resource(&normalized)
                {
                    let installed = decode_bytes(&bytes, self.schema)?;
                    resolve_embedded_layer_textures(&mut document, &installed, |path| {
                        self.archives.contains_resource_hash(resource_hash(path))
                    })?;
                }
                if extension == "mlsetup" {
                    resolve_microblend_defaults(&mut document, |path| {
                        self.archives.contains_resource_hash(resource_hash(path))
                            || self
                                .embedded_resources
                                .contains_key(&path.replace('/', "\\").to_lowercase())
                    })?;
                }
                match extension {
                    "xbm" => {
                        export_xbm(&document, &output)?;
                        self.textures.insert(normalized.clone());
                        remember_dependency(
                            &resource_lock,
                            output.with_extension("png"),
                            Vec::new(),
                        )?;
                    }
                    "mlmask" => {
                        export_mlmask(&document, &output)?;
                        self.masks += 1;
                        remember_dependency(
                            &resource_lock,
                            output.with_extension("masklist"),
                            Vec::new(),
                        )?;
                    }
                    _ => {
                        write_json(&json_output, &document)?;
                        let mut dependencies = VecDeque::new();
                        collect_depot_paths(&document, &mut dependencies);
                        remember_dependency(
                            &resource_lock,
                            json_output,
                            dependencies.iter().cloned().collect(),
                        )?;
                        queue.extend(dependencies);
                    }
                }
                Ok(())
            })();
            if let Err(source) = result {
                let error = MaterialError::Dependency {
                    path: normalized,
                    source: Box::new(source),
                };
                if self.defer_errors {
                    // Instances can retain unused shader parameters pointing
                    // at resources absent from the shipped game. PBR's strict
                    // texture/resource readers still reject any missing input
                    // actually consumed by the selected shader.
                    self.errors.push(error.to_string());
                } else {
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

fn dependency_lock(path: &Path) -> Arc<DependencySlot> {
    let mut locks = DEPENDENCY_LOCKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Arc::clone(
        locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(DependencySlot::default())),
    )
}

fn remember_dependency(
    slot: &DependencySlot,
    artifact: PathBuf,
    dependencies: Vec<String>,
) -> Result<(), MaterialError> {
    if slot.completed.get().is_none() {
        let metadata = fs::metadata(&artifact)?;
        let _ = slot.completed.set(ExportedDependency {
            artifact,
            size: metadata.len(),
            modified: metadata.modified()?,
            dependencies,
        });
    }
    Ok(())
}

fn export_xbm(document: &Value, output: &Path) -> Result<(), MaterialError> {
    let root = path(document, &["Data", "RootChunk"])?;
    // STextureGroupSetup leaves compression at enum zero (TCM_None).
    // The native reader omits constructor-default properties.
    let compression = path(root, &["setup"])?
        .get("compression")
        .and_then(Value::as_str)
        .unwrap_or("TCM_None");
    let blob = path(
        root,
        &["renderTextureResource", "renderResourceBlobPC", "Data"],
    )?;
    let header = path(blob, &["header"])?;
    let size_info = path(header, &["sizeInfo"])?;
    let width = required_usize(size_info, "width")?;
    let height = required_usize(size_info, "height")?;
    let mip = array(header, "mipMapInfo")?
        .first()
        .ok_or_else(|| malformed("texture has no mipmaps"))?;
    let placement = path(mip, &["placement"])?;
    let offset = usize_value(placement.get("offset").unwrap_or(&Value::Null)).unwrap_or(0);
    let size = usize_value(
        placement
            .get("size")
            .ok_or_else(|| malformed("texture mip has no size"))?,
    )
    .ok_or_else(|| malformed("texture mip size is invalid"))?;
    let encoded = string(path(blob, &["textureData"])?, "Bytes")?;
    let bytes = STANDARD.decode(encoded)?;
    let end = offset
        .checked_add(size)
        .ok_or_else(|| malformed("texture mip range overflows"))?;
    let data = bytes
        .get(offset..end)
        .ok_or_else(|| malformed("texture mip range is outside its buffer"))?;

    let mut pixels = vec![0_u32; width * height];
    match compression {
        "TCM_DXTNoAlpha" | "TCM_Normals_DEPRECATED" => decode_bc1(data, width, height, &mut pixels),
        "TCM_DXTAlpha" | "TCM_DXTAlphaLinear" | "TCM_NormalsHigh_DEPRECATED" => {
            decode_bc3(data, width, height, &mut pixels)
        }
        "TCM_Normalmap" | "TCM_QualityRG" => decode_bc5(data, width, height, &mut pixels),
        "TCM_QualityR" => decode_bc4(data, width, height, &mut pixels),
        "TCM_QualityColor" => decode_bc7(data, width, height, &mut pixels),
        "TCM_HalfHDR" | "TCM_HalfHDR_Unsigned" => {
            decode_bc6_unsigned(data, width, height, &mut pixels)
        }
        "TCM_HalfHDRSigned" => decode_bc6_signed(data, width, height, &mut pixels),
        "TCM_None" => {
            return export_uncompressed_texture(root, data, width, height, output);
        }
        other => return Err(MaterialError::UnsupportedTexture(other.to_owned())),
    }
    .map_err(|error| {
        MaterialError::Texture(format!(
            "{error} ({compression}, {width}x{height}, {} bytes)",
            data.len()
        ))
    })?;

    let grayscale = compression == "TCM_QualityR";
    let mut rgba = Vec::with_capacity(pixels.len() * 4);
    for row in pixels.rchunks_exact(width) {
        for pixel in row {
            let [blue, green, red, alpha] = pixel.to_le_bytes();
            if grayscale {
                rgba.extend_from_slice(&[red, red, red, alpha]);
            } else {
                rgba.extend_from_slice(&[red, green, blue, alpha]);
            }
        }
    }
    write_png(output.with_extension("png").as_path(), width, height, &rgba)
}

fn export_uncompressed_texture(
    root: &Value,
    data: &[u8],
    width: usize,
    height: usize,
    output: &Path,
) -> Result<(), MaterialError> {
    let raw_format = root
        .get("setup")
        .and_then(|setup| setup.get("rawFormat"))
        .and_then(Value::as_str)
        .unwrap_or("TRF_TrueColor");
    let rgba = match raw_format {
        "TRF_Grayscale" if data.len() >= width * height => data[..width * height]
            .iter()
            .flat_map(|value| [*value, *value, *value, 255])
            .collect(),
        _ if data.len() >= width * height * 4 => data[..width * height * 4].to_vec(),
        _ => return Err(malformed("uncompressed texture payload is too small")),
    };
    write_png(output.with_extension("png").as_path(), width, height, &rgba)
}

fn export_mlmask(document: &Value, output: &Path) -> Result<(), MaterialError> {
    let blob = path(
        document,
        &[
            "Data",
            "RootChunk",
            "renderResourceBlob",
            "renderResourceBlobPC",
            "Data",
        ],
    )?;
    let header = path(blob, &["header"])?;
    let atlas_width = required_usize(header, "atlasWidth")?;
    let atlas_height = required_usize(header, "atlasHeight")?;
    let mask_count = required_usize(header, "numLayers")?;
    let mask_width = required_usize(header, "maskWidth")?;
    let mask_height = required_usize(header, "maskHeight")?;
    // Zero-valued reflected fields are omitted. Small masks have no low mip.
    let low_width = header
        .get("maskWidthLow")
        .and_then(usize_value)
        .unwrap_or(0);
    let low_height = header
        .get("maskHeightLow")
        .and_then(usize_value)
        .unwrap_or(0);
    let tile_size = required_usize(header, "maskTileSize")?;
    if tile_size == 0 {
        return Err(malformed("multilayer mask tile size is zero"));
    }
    let atlas_bytes = STANDARD.decode(string(path(blob, &["atlasData"])?, "Bytes")?)?;
    let mut atlas_pixels = vec![0_u32; atlas_width * atlas_height];
    decode_bc4(&atlas_bytes, atlas_width, atlas_height, &mut atlas_pixels)
        .map_err(|error| MaterialError::Texture(error.to_owned()))?;
    let atlas: Vec<u8> = atlas_pixels
        .into_iter()
        .map(|pixel| pixel.to_le_bytes()[2])
        .collect();
    let tile_bytes = STANDARD.decode(string(path(blob, &["tilesData"])?, "Bytes")?)?;
    let tiles: Vec<u32> = tile_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();

    let stem = output
        .file_stem()
        .and_then(OsStr::to_str)
        .ok_or_else(|| malformed("multilayer mask output has no file stem"))?;
    let parent = output
        .parent()
        .ok_or_else(|| malformed("multilayer mask output has no parent"))?;
    let layer_directory = parent.join(format!("{stem}_layers"));
    fs::create_dir_all(&layer_directory)?;
    let mut mask_list = String::new();
    for layer_index in 0..mask_count {
        let full = decode_mask_layer(
            &atlas,
            atlas_width,
            &tiles,
            mask_width,
            mask_height,
            low_width,
            tile_size,
            layer_index,
        );
        let high_resolution =
            has_high_resolution(&tiles, mask_width, mask_height, tile_size, layer_index);
        let (pixels, width, height) =
            if high_resolution || low_width == 0 || low_height == 0 || low_width == mask_width {
                (full, mask_width, mask_height)
            } else {
                (
                    downscale_nearest(&full, mask_width, mask_height, low_width, low_height),
                    low_width,
                    low_height,
                )
            };
        let rgba: Vec<u8> = pixels
            .iter()
            .flat_map(|value| [*value, *value, *value, 255])
            .collect();
        let name = format!("{stem}_{layer_index}.png");
        write_png(&layer_directory.join(&name), width, height, &rgba)?;
        writeln!(mask_list, "{stem}_layers/{name}")
            .expect("writing a multilayer mask path to a String cannot fail");
    }
    fs::write(output.with_extension("masklist"), mask_list)?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the dimensions mirror the independently versioned mask header fields"
)]
fn decode_mask_layer(
    atlas: &[u8],
    atlas_width: usize,
    tiles: &[u32],
    mask_width: usize,
    mask_height: usize,
    low_width: usize,
    tile_size: usize,
    layer_index: usize,
) -> Vec<u8> {
    let mut output = vec![0_u8; mask_width * mask_height];
    let full_width_tiles = mask_width.div_ceil(tile_size);
    let full_height_tiles = mask_height.div_ceil(tile_size);
    let low_offset = full_width_tiles * full_height_tiles;
    let small_scale = if low_width == 0 || mask_width < low_width {
        1
    } else {
        mask_width / low_width
    };
    let low_width_tiles = (mask_width / small_scale).div_ceil(tile_size);
    for y in 0..mask_height {
        for x in 0..mask_width {
            if !decode_mask_pixel(
                atlas,
                atlas_width,
                tiles,
                x,
                y,
                layer_index,
                0,
                1,
                full_width_tiles,
                tile_size,
                &mut output,
                mask_width,
            ) {
                decode_mask_pixel(
                    atlas,
                    atlas_width,
                    tiles,
                    x,
                    y,
                    layer_index,
                    low_offset,
                    small_scale,
                    low_width_tiles,
                    tile_size,
                    &mut output,
                    mask_width,
                );
            }
        }
    }
    output
}

#[expect(
    clippy::too_many_arguments,
    reason = "this is the direct tile lookup operation defined by the mask format"
)]
fn decode_mask_pixel(
    atlas: &[u8],
    atlas_width: usize,
    tiles: &[u32],
    x: usize,
    y: usize,
    layer_index: usize,
    tiles_offset: usize,
    small_scale: usize,
    width_in_tiles: usize,
    tile_size: usize,
    output: &mut [u8],
    mask_width: usize,
) -> bool {
    let tile_index = width_in_tiles * (y / tile_size / small_scale)
        + (x / tile_size / small_scale)
        + tiles_offset;
    let Some((&parameter_offset, &parameter_bits)) =
        tiles.get(tile_index * 2).zip(tiles.get(tile_index * 2 + 1))
    else {
        return false;
    };
    let layer_bit = 1_u32.checked_shl(u32::try_from(layer_index).unwrap_or(u32::MAX));
    let Some(layer_bit) = layer_bit else {
        return false;
    };
    if parameter_bits & layer_bit == 0 {
        return false;
    }
    let preceding = parameter_bits & layer_bit.saturating_sub(1);
    let declaration_index =
        usize::try_from(parameter_offset).unwrap_or(usize::MAX) + preceding.count_ones() as usize;
    let Some(&declaration) = tiles.get(declaration_index) else {
        return false;
    };
    let dx = usize::try_from(declaration & 0x3ff).unwrap_or(0);
    let dy = usize::try_from((declaration >> 10) & 0x3ff).unwrap_or(0);
    let sx = usize::try_from((declaration >> 20) & 0xf).unwrap_or(0);
    let sy = usize::try_from((declaration >> 24) & 0xf).unwrap_or(0);
    let local_x = ((x >> sx) % tile_size).min(tile_size - 1);
    let local_y = ((y >> sy) % tile_size).min(tile_size - 1);
    let atlas_tile_size = tile_size + 2;
    let atlas_index =
        local_x + 1 + dx * atlas_tile_size + (local_y + 1 + dy * atlas_tile_size) * atlas_width;
    let Some(&pixel) = atlas.get(atlas_index) else {
        return false;
    };
    if let Some(target) = output.get_mut(x + y * mask_width) {
        *target = pixel;
        true
    } else {
        false
    }
}

fn has_high_resolution(
    tiles: &[u32],
    width: usize,
    height: usize,
    tile_size: usize,
    layer_index: usize,
) -> bool {
    let Some(layer_bit) = 1_u32.checked_shl(u32::try_from(layer_index).unwrap_or(u32::MAX)) else {
        return false;
    };
    (0..width.div_ceil(tile_size) * height.div_ceil(tile_size)).any(|index| {
        tiles
            .get(index * 2 + 1)
            .is_some_and(|bits| bits & layer_bit != 0)
    })
}

fn downscale_nearest(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut output = vec![0_u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x * source_width / width).min(source_width - 1);
            let source_y = (y * source_height / height).min(source_height - 1);
            output[x + y * width] = source[source_x + source_y * source_width];
        }
    }
    output
}

fn write_png(path: &Path, width: usize, height: usize, rgba: &[u8]) -> Result<(), MaterialError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let mut encoder = png::Encoder::new(
        file,
        u32::try_from(width).map_err(|_| malformed("PNG width does not fit u32"))?,
        u32::try_from(height).map_err(|_| malformed("PNG height does not fit u32"))?,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder
        .write_header()
        .map_err(|error| io::Error::other(error.to_string()))?
        .write_image_data(rgba)
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

fn decode_bytes(bytes: &[u8], schema: &RedSchema) -> Result<Value, MaterialError> {
    Ok(codec::decode_wkit_from_bytes(bytes, schema)?)
}

fn collect_depot_paths(value: &Value, queue: &mut VecDeque<String>) {
    match value {
        Value::Object(object) => {
            if let Some(path) = object
                .get("DepotPath")
                .and_then(|value| value.get("$value"))
                .and_then(Value::as_str)
            {
                queue.push_back(path.to_owned());
            }
            for value in object.values() {
                collect_depot_paths(value, queue);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_depot_paths(value, queue);
            }
        }
        Value::String(value)
            if extension(value).is_some_and(|ext| DEPOT_EXTENSIONS.contains(&ext)) =>
        {
            queue.push_back(value.to_owned());
        }
        _ => {}
    }
}

fn normalize_value(value: &Value) -> Value {
    if let Ok(path) = red_depot_path(value) {
        return Value::String(path);
    }
    match value {
        Value::Object(object) => {
            if let Some(value) = object.get("$value") {
                return normalize_value(value);
            }
            let mut normalized: Map<String, Value> = object
                .iter()
                .filter(|(key, _)| !key.starts_with('$') && *key != "Flags")
                .map(|(key, value)| (key.clone(), normalize_value(value)))
                .collect();
            let fields: &[&str] = match object.get("$type").and_then(Value::as_str) {
                Some("Vector4") => &["X", "Y", "Z", "W"],
                Some("Color" | "CColor") => &["Red", "Green", "Blue", "Alpha"],
                _ => &[],
            };
            for field in fields {
                normalized.entry((*field).to_owned()).or_insert(json!(0));
            }
            Value::Object(normalized)
        }
        Value::Array(values) => Value::Array(values.iter().map(normalize_value).collect()),
        other => other.clone(),
    }
}

fn complete_blender_defaults(document: &mut Value, extension: &str) -> Result<(), MaterialError> {
    if let Some(header) = document.get_mut("Header").and_then(Value::as_object_mut) {
        header.insert(
            "WolvenKitVersion".to_owned(),
            json!("8.17-compatible (ghostline-red 0.1.0)"),
        );
    }
    let root = document
        .get_mut("Data")
        .and_then(|value| value.get_mut("RootChunk"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| malformed("resource JSON has no root chunk"))?;
    match extension {
        "mlsetup" => {
            root.entry("ratio").or_insert(json!(1));
            root.entry("useNormal").or_insert(json!(1));
            if let Some(layers) = root.get_mut("layers").and_then(Value::as_array_mut) {
                for layer in layers {
                    let layer = layer
                        .as_object_mut()
                        .ok_or_else(|| malformed("multilayer setup layer is not an object"))?;
                    layer.entry("matTile").or_insert(json!(1));
                    layer.entry("mbTile").or_insert(json!(1));
                    layer.entry("microblendContrast").or_insert(json!(0.5));
                    layer.entry("microblendOffsetU").or_insert(json!(0));
                    layer.entry("microblendOffsetV").or_insert(json!(0));
                    layer.entry("offsetU").or_insert(json!(0));
                    layer.entry("offsetV").or_insert(json!(0));
                    layer.entry("opacity").or_insert(json!(1));
                    layer.entry("overrides").or_insert(json!({
                        "$type": "CName",
                        "$storage": "string",
                        "$value": "None"
                    }));
                }
            }
        }
        "mltemplate" => {
            root.entry("colorMaskLevelsOut")
                .or_insert(json!({"Elements": [0, 0]}));
            root.entry("tilingMultiplier").or_insert(json!(1));
            root.entry("defaultOverrides").or_insert(json!({
                "$type": "Multilayer_LayerOverrideSelection",
                "colorScale": red_name("null_null"),
                "metalLevelsIn": red_name("null"),
                "metalLevelsOut": red_name("null"),
                "normalStrength": red_name("null"),
                "roughLevelsIn": red_name("null"),
                "roughLevelsOut": red_name("null")
            }));
        }
        _ => {}
    }
    Ok(())
}

fn resolve_embedded_layer_textures(
    document: &mut Value,
    installed: &Value,
    exists: impl Fn(&str) -> bool,
) -> Result<(), MaterialError> {
    let native = path(installed, &["Data", "RootChunk"])?;
    let root = document
        .get_mut("Data")
        .and_then(|v| v.get_mut("RootChunk"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| malformed("layer template has no root"))?;
    let mut fallbacks = Vec::new();
    for key in [
        "colorTexture",
        "normalTexture",
        "roughnessTexture",
        "metalnessTexture",
    ] {
        let Some(missing) = root.get(key).and_then(|v| red_depot_path(v).ok()) else {
            continue;
        };
        if exists(&missing) {
            continue;
        }
        let Some(replacement) = native.get(key) else {
            continue;
        };
        let target = red_depot_path(replacement)?;
        if exists(&target) {
            root.insert(key.into(), replacement.clone());
            fallbacks.push(json!({"parameter":key,"missing":missing,"replacement":target,
                "reason":"missing embedded layer input replaced by the same installed native layer template's input"}));
        }
    }
    if !fallbacks.is_empty() {
        root.insert("ResourceFallbacks".into(), json!(fallbacks));
    }
    Ok(())
}

fn red_name(value: &str) -> Value {
    json!({"$type": "CName", "$storage": "string", "$value": value})
}

// The native multilayer_default.mlsetup declares this neutral microblend on its
// layers. Preserve an unresolved authored reference in the generated repository's
// diagnostics, while retaining all of that layer's color/material/UV settings.
fn resolve_microblend_defaults(
    document: &mut Value,
    exists: impl Fn(&str) -> bool,
) -> Result<(), MaterialError> {
    let fallback = r"engine\materials\defaults\ml_default_microblend.xbm";
    if !exists(fallback) {
        return Ok(());
    }
    let root = document
        .get_mut("Data")
        .and_then(|data| data.get_mut("RootChunk"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| malformed("multilayer setup has no root"))?;
    let mut fallbacks = Vec::new();
    for layer in root
        .get_mut("layers")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        let Some(value) = layer.get_mut("microblend") else {
            continue;
        };
        let requested = red_depot_path(value)?;
        if !requested.is_empty() && !exists(&requested) {
            fallbacks.push(json!({"parameter":"microblend","missing":requested,"replacement":fallback,
                "reason":"Unresolved microblend; neutral input declared by native multilayer_default.mlsetup"}));
            *value = json!({"DepotPath":{"$type":"ResourcePath","$storage":"string","$value":fallback},"Flags":"Default"});
        }
    }
    if !fallbacks.is_empty() {
        root.insert("ResourceFallbacks".into(), json!(fallbacks));
    }
    Ok(())
}

#[test]
fn missing_microblend_uses_neutral_input_and_retains_source_diagnostic() {
    let fallback = r"engine\materials\defaults\ml_default_microblend.xbm";
    let mut document = json!({"Data":{"RootChunk":{"layers":[
        {"material":"paint.mltemplate","microblend":{"DepotPath":{"$value":"missing.xbm"}},"opacity":0.75},
        {"microblend":{"DepotPath":{"$value":"present.xbm"}}}
    ]}}});
    resolve_microblend_defaults(&mut document, |path| {
        path == fallback || path == "present.xbm"
    })
    .unwrap();
    let root = &document["Data"]["RootChunk"];
    assert_eq!(
        red_depot_path(&root["layers"][0]["microblend"]).unwrap(),
        fallback
    );
    assert_eq!(root["layers"][0]["opacity"], 0.75);
    assert_eq!(root["ResourceFallbacks"][0]["missing"], "missing.xbm");
    assert_eq!(
        red_depot_path(&root["layers"][1]["microblend"]).unwrap(),
        "present.xbm"
    );
}

fn template_defaults(template: &str) -> Value {
    if template.ends_with("multilayered.mt") {
        json!({
            "GlobalNormal": "engine\\textures\\editor\\normal.xbm",
            "MultilayerMask": "engine\\materials\\defaults\\multilayer_default.mlmask",
            "MultilayerSetup": "engine\\materials\\defaults\\multilayer_default.mlsetup",
            "GlobalNormalIntensity": 1,
            "GlobalNormalUVScale": {"X": 1, "Y": 1, "Z": 0, "W": 0},
            "GlobalNormalUVBias": {"X": 0, "Y": 0, "Z": 0, "W": 0},
            "MaskAtlas": "",
            "MaskTiles": null,
            "Layers": null,
            "LayersStartIndex": 0,
            "SurfaceTexAspectRatio": 0,
            "MaskToTileScale": {"X": 0, "Y": 0, "Z": 0, "W": 0},
            "MaskTileSize": 0,
            "MaskAtlasDims": {"X": 0, "Y": 0, "Z": 0, "W": 0},
            "MaskBaseResolution": {"X": 0, "Y": 0, "Z": 0, "W": 0},
            "SetupLayerMask": 0,
            "NormalsTextureDDXYMultiplier": 1,
            "MicroblendsTextureDDXYMultiplier": 1
        })
    } else {
        json!({})
    }
}

fn repo_path(repository: &Path, depot_path: &str) -> Result<PathBuf, MaterialError> {
    let mut result = repository.to_path_buf();
    for component in Path::new(&depot_path.replace('\\', "/")).components() {
        match component {
            Component::Normal(value) => result.push(value),
            _ => return Err(malformed("depot path is not relative and normalized")),
        }
    }
    Ok(result)
}

fn write_json(output: &Path, document: &Value) -> Result<(), MaterialError> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(document)?)?;
    Ok(())
}

fn red_depot_path(value: &Value) -> Result<String, MaterialError> {
    value
        .get("DepotPath")
        .and_then(|value| value.get("$value"))
        .or_else(|| value.get("$value"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| malformed("resource reference has no depot path"))
}

fn red_string(value: &Value) -> Result<String, MaterialError> {
    value
        .get("$value")
        .and_then(Value::as_str)
        .or_else(|| value.as_str())
        .map(str::to_owned)
        .ok_or_else(|| malformed("RED string value is missing"))
}

fn path<'a>(value: &'a Value, members: &[&str]) -> Result<&'a Value, MaterialError> {
    members.iter().try_fold(value, |current, member| {
        current
            .get(*member)
            .ok_or_else(|| malformed("required object member is missing"))
    })
}

fn array<'a>(value: &'a Value, member: &str) -> Result<&'a Vec<Value>, MaterialError> {
    value
        .get(member)
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("required array member is missing"))
}

fn string<'a>(value: &'a Value, member: &str) -> Result<&'a str, MaterialError> {
    value
        .get(member)
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("required string member is missing"))
}

fn required_usize(value: &Value, member: &str) -> Result<usize, MaterialError> {
    value
        .get(member)
        .and_then(usize_value)
        .ok_or_else(|| malformed("required integer member is missing"))
}

fn usize_value(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

fn extension(path: &str) -> Option<&str> {
    path.rsplit_once('.').map(|(_, extension)| extension)
}

fn malformed(message: impl Into<String>) -> MaterialError {
    MaterialError::Malformed(message.into())
}

#[cfg(test)]
mod tests {
    use super::{downscale_nearest, export_xbm, normalize_value, repo_path};
    use base64::Engine as _;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn hash_addressed_resource_identity_preserves_all_64_bits() {
        let hash = 12_343_119_027_633_579_865_u64;
        assert_eq!(
            super::resource_hash("__hash__/12343119027633579865.mesh"),
            hash
        );
        assert_eq!(
            super::resource_hash(r"__hash__\12343119027633579865.mesh"),
            hash
        );
        assert_eq!(
            super::resource_hash(r"base\known.mesh"),
            crate::archive::depot_path_hash(r"base\known.mesh")
        );
    }

    #[test]
    fn missing_normal_uses_only_an_available_template_default_and_retains_warning() {
        let folder = tempfile::tempdir().unwrap();
        let mut archives = super::ArchiveSet::open(folder.path()).unwrap();
        let template = r"engine\materials\test.mt";
        let flat = r"engine\textures\flat.xbm";
        archives
            .locations
            .insert(super::resource_hash(flat), (0, 0));
        archives.template_cache.lock().unwrap().insert(
            template.to_owned(),
            serde_json::Map::from_iter([("Normal".to_owned(), json!(flat))]),
        );
        let document = json!({"Data":{"RootChunk":{"baseMaterial":{"DepotPath":{"$value":template}},"values":[{"Normal":{"DepotPath":{"$value":"missing\\normal.xbm"}}},{"BaseColor":{"DepotPath":{"$value":"missing\\paint.xbm"}}}]}}});
        let record = super::resolve_material(
            "body".to_owned(),
            &document,
            &archives,
            &crate::schema::RedSchema::default(),
            0,
            &mut std::collections::HashMap::new(),
        )
        .unwrap();
        assert_eq!(record.data["Normal"], flat);
        assert_eq!(record.data["BaseColor"], r"missing\paint.xbm");
        assert_eq!(record.resource_fallbacks.len(), 1);
    }

    #[test]
    fn template_parameters_supply_resources_and_omitted_scalar_zero() {
        let parameters = json!({"Elements":[[],[{"Data":{"$type":"CMaterialParameterMultilayerMask","parameterName":{"$value":"MultilayerMask"},"mask":{"DepotPath":{"$value":"engine\\materials\\defaults\\multilayer_default.mlmask"}}}}, {"Data":{"$type":"CMaterialParameterScalar","parameterName":{"$value":"NormalStrength"}}}]]});
        let mut values = serde_json::Map::new();
        super::collect_template_values(&parameters, &mut values);
        assert_eq!(
            values["MultilayerMask"],
            r"engine\materials\defaults\multilayer_default.mlmask"
        );
        assert_eq!(values["NormalStrength"], 0.0);
    }

    #[test]
    fn embedded_material_inheritance_retains_parent_owned_dependencies() {
        let folder = tempfile::tempdir().unwrap();
        let archives = super::ArchiveSet::open(folder.path()).unwrap();
        let template = r"engine\test.mt";
        archives
            .template_cache
            .lock()
            .unwrap()
            .insert(template.into(), serde_json::Map::new());
        let setup = r"ep1\vehicle\paint.mlsetup";
        let parent = json!({"Data":{"RootChunk":{"baseMaterial":{"DepotPath":{"$value":template}},"values":[{"MultilayerSetup":{"DepotPath":{"$value":setup},"Flags":"Embedded"}}]},"EmbeddedFiles":[{"FileName":{"$value":setup},"Content":{"$type":"Multilayer_Setup","layers":[]}}]}});
        let mut embedded =
            std::collections::HashMap::from([(r"ep1\vehicle\paint.mi".into(), parent)]);
        let child = json!({"Data":{"RootChunk":{"baseMaterial":{"DepotPath":{"$value":r"ep1\vehicle\paint.mi"},"Flags":"Embedded"},"values":[]}}});
        let material = super::resolve_material(
            "paint".into(),
            &child,
            &archives,
            &crate::schema::RedSchema::default(),
            0,
            &mut embedded,
        )
        .unwrap();
        assert_eq!(material.material_template, template);
        assert_eq!(material.data["MultilayerSetup"], setup);
        assert_eq!(
            embedded[setup]["Data"]["RootChunk"]["$type"],
            "Multilayer_Setup"
        );
    }

    #[test]
    fn omitted_name_after_a_named_appearance_is_still_default() {
        let root = json!({"appearances":[{"Data":{"name":{"$value":"red"},"chunkMaterials":[{"$value":"paint"}]}},{"Data":{"chunkMaterials":[{"$value":"base"}]}}]});
        let decoded = super::decode_appearances(&root).unwrap();
        assert_eq!(decoded["default"], json!(["base"]));
        assert!(!decoded.contains_key("default1"));
    }

    #[test]
    fn deferred_dependency_errors_remain_recorded_and_strict_mode_rejects_them() {
        let folder = tempfile::tempdir().unwrap();
        let archives = super::ArchiveSet::open(folder.path()).unwrap();
        let schema = crate::schema::RedSchema::default();
        let embedded = std::collections::HashMap::new();
        let mut exporter =
            super::DependencyExporter::new(&archives, &schema, folder.path(), &embedded);
        let queue = std::collections::VecDeque::from([r"missing\unused.xbm".to_owned()]);
        assert!(exporter.export(queue.clone()).is_err());
        exporter.visited.clear();
        exporter.defer_errors = true;
        exporter.export(queue).unwrap();
        assert_eq!(exporter.errors.len(), 1);
        assert!(exporter.errors[0].contains("was not found"));
    }

    #[test]
    fn omitted_texture_compression_decodes_transparent_pixel() {
        let folder = tempfile::tempdir().unwrap();
        let output = folder.path().join("transparent.xbm");
        let document = json!({"Data":{"RootChunk":{"setup":{},"renderTextureResource":{"renderResourceBlobPC":{"Data":{
            "header":{"sizeInfo":{"width":1,"height":1},"mipMapInfo":[{"placement":{"size":4}}]},
            "textureData":{"Bytes":base64::engine::general_purpose::STANDARD.encode([0_u8;4])}
        }}}}}});
        export_xbm(&document, &output).unwrap();
        let image = image::open(output.with_extension("png"))
            .unwrap()
            .into_rgba8();
        assert_eq!(image.get_pixel(0, 0).0, [0, 0, 0, 0]);
    }

    #[test]
    fn normalizes_red_values_for_material_sidecars() {
        let value = json!({
            "$type": "rRef:ITexture",
            "DepotPath": {
                "$type": "ResourcePath",
                "$storage": "string",
                "$value": "base\\texture.xbm"
            },
            "Flags": "Default"
        });
        assert_eq!(normalize_value(&value), json!("base\\texture.xbm"));
    }

    #[test]
    fn preserves_omitted_zero_material_vector_channels() {
        assert_eq!(
            normalize_value(&json!({"$type":"Vector4","W":1.0})),
            json!({"X":0,"Y":0,"Z":0,"W":1.0})
        );
        assert_eq!(
            normalize_value(&json!({"$type":"Color","Green":128})),
            json!({"Red":0,"Green":128,"Blue":0,"Alpha":0})
        );
    }

    #[test]
    fn rejects_escaping_depot_paths() {
        assert!(repo_path(Path::new("repo"), "..\\outside.xbm").is_err());
    }

    #[test]
    fn nearest_downscale_samples_source_grid() {
        assert_eq!(
            downscale_nearest(&[0, 1, 2, 3, 4, 5, 6, 7, 8], 3, 3, 2, 2),
            vec![0, 1, 3, 4]
        );
    }
}
