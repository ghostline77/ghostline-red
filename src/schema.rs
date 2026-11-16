//! RED reflection with bundled serialization definitions and game metadata overlays.

mod bundle;
mod scripts;
pub use bundle::{Provenance, Source};

use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::Path,
    sync::OnceLock,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("could not access schema source: {0}")]
    Io(#[from] io::Error),
    #[error("invalid RED reflection schema: {0}")]
    Invalid(String),
    #[error("could not read or encode schema JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedClass {
    pub base: Option<String>,
    pub properties: BTreeMap<String, RedProperty>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flags: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedProperty {
    pub cs_type: String,
    pub ordinal: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub red_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flags: Option<u64>,
}

/// Holds RED reflection types from legacy schemas or official `REDmod` metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RedSchema {
    pub classes: BTreeMap<String, RedClass>,
    pub enums: BTreeSet<String>,
    pub bitfields: BTreeSet<String>,
    pub aliases: BTreeMap<String, String>,
    pub simple_types: BTreeSet<String>,
}

impl RedSchema {
    /// Loads the bundled baseline, installed game metadata and an optional override.
    ///
    /// Installed native metadata takes precedence over the baseline. An explicit
    /// override is applied last. No game installation is needed for offline files.
    ///
    /// # Errors
    /// Returns an error for malformed installed metadata or an unreadable override.
    pub fn load(
        game_root: Option<&Path>,
        override_path: Option<&Path>,
    ) -> Result<Self, SchemaError> {
        let mut schema = Self::bundled()?;
        if let Some(game) = game_root {
            let scripts = game.join("r6/cache/final.redscripts");
            if scripts.is_file() {
                let mut installed = self::scripts::read(&fs::read(scripts)?)?;
                // Script identifiers are not always serialized property names.
                // Retain the established serialization contract for known fields.
                installed.merge(schema);
                schema = installed;
            }
            let native = game.join("tools/redmod/metadata.json");
            if native.is_file() {
                schema.merge(Self::from_slice(&fs::read(native)?)?);
            }
        }
        if let Some(path) = override_path {
            schema.merge(Self::from_slice(&fs::read(path)?)?);
        }
        Ok(schema)
    }

    /// Loads the embedded serialization contract and game snapshot without filesystem access.
    ///
    /// # Errors
    /// Returns an error if the embedded snapshot is invalid.
    pub fn bundled() -> Result<Self, SchemaError> {
        static BUNDLED: OnceLock<Result<RedSchema, String>> = OnceLock::new();
        BUNDLED
            .get_or_init(|| {
                Self::from_slice(include_bytes!("../data/schema.json"))
                    .and_then(|schema| {
                        if schema.classes.is_empty() {
                            return Err(SchemaError::Invalid(
                                "bundled schema has no classes".into(),
                            ));
                        }
                        let native = Self {
                            classes: schema
                                .classes
                                .iter()
                                .filter(|(_, class)| class.flags.is_some())
                                .map(|(name, class)| (name.clone(), class.clone()))
                                .collect(),
                            ..Self::default()
                        };
                        let mut result = schema;
                        result.merge(Self::from_slice(include_bytes!(
                            "../data/serialization.json"
                        ))?);
                        result.merge(native);
                        Ok(result)
                    })
                    .map_err(|error| error.to_string())
            })
            .clone()
            .map_err(SchemaError::Invalid)
    }

    /// Returns reproducible fingerprints for the game snapshot and serialization definitions.
    ///
    /// # Errors
    /// Returns an error if the embedded provenance is invalid.
    pub fn bundled_provenance() -> Result<Provenance, SchemaError> {
        let document: bundle::Document =
            serde_json::from_slice(include_bytes!("../data/schema.json"))?;
        let mut provenance = document.validate()?.provenance;
        provenance.sources.push(Source {
            path: "ghostline-red/data/serialization.json".into(),
            sha1: format!(
                "{:x}",
                Sha1::digest(include_bytes!("../data/serialization.json"))
            ),
        });
        Ok(provenance)
    }

    /// Loads installed game metadata and an explicitly supplied schema override.
    ///
    /// # Errors
    /// Returns an error for unreadable or malformed metadata.
    pub fn from_game_files(game_root: &Path, schema_path: &Path) -> Result<Self, SchemaError> {
        Self::load(Some(game_root), Some(schema_path))
    }
    /// Parses either Ghostline's legacy class map or `REDmod` RTTI metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::Json`] when `bytes` are not a supported JSON
    /// schema document.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, SchemaError> {
        let document: SchemaDocument = serde_json::from_slice(bytes)?;
        Ok(match document {
            SchemaDocument::Legacy(classes) => {
                let mut result = Self {
                    classes,
                    ..Self::default()
                };
                for class in result.classes.values() {
                    for property in class.properties.values() {
                        for (prefix, types) in [
                            ("CEnum<", &mut result.enums),
                            ("CBitField<", &mut result.bitfields),
                        ] {
                            if let Some((_, suffix)) = property.cs_type.split_once(prefix)
                                && let Some((name, _)) = suffix.split_once('>')
                            {
                                types.insert(name.into());
                            }
                        }
                    }
                }
                result
            }
            SchemaDocument::Redmod(metadata) => metadata.into(),
            SchemaDocument::Bundled(document) => document.validate()?.schema,
        })
    }

    /// Merges another schema into this one.
    ///
    /// Classes and properties absent from `other` are retained. Metadata
    /// present in `other` takes precedence, so callers can load a broad
    /// compatibility schema first and overlay official `REDmod` metadata.
    pub fn merge(&mut self, other: Self) {
        for (name, incoming) in other.classes {
            self.classes
                .entry(name)
                .and_modify(|class| {
                    if incoming.base.is_some() {
                        class.base.clone_from(&incoming.base);
                    }
                    class.properties.extend(incoming.properties.clone());
                    if incoming.flags.is_some() {
                        class.flags = incoming.flags;
                    }
                    if incoming.size.is_some() {
                        class.size = incoming.size;
                    }
                    if incoming.alignment.is_some() {
                        class.alignment = incoming.alignment;
                    }
                })
                .or_insert(incoming);
        }
        self.enums.extend(other.enums);
        self.bitfields.extend(other.bitfields);
        self.aliases.extend(other.aliases);
        self.simple_types.extend(other.simple_types);
    }

    /// Returns every reflected class name.
    #[must_use]
    pub fn class_names(&self) -> BTreeSet<String> {
        let mut names = self.classes.keys().cloned().collect::<BTreeSet<_>>();
        for alias in self.aliases.keys() {
            let mut target = alias.as_str();
            let mut visited = BTreeSet::new();
            while let Some(next) = self.aliases.get(target) {
                if !visited.insert(target) {
                    break;
                }
                target = next;
            }
            if self.classes.contains_key(target) {
                names.insert(alias.clone());
            }
        }
        names
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SchemaDocument {
    Bundled(bundle::Document),
    Redmod(RedmodMetadata),
    Legacy(BTreeMap<String, RedClass>),
}

#[derive(Debug, Deserialize)]
struct RedmodMetadata {
    #[serde(default)]
    simple_types: Vec<RedmodNamedType>,
    classes: Vec<RedmodClass>,
    #[serde(default)]
    enums: Vec<RedmodNamedType>,
    #[serde(default)]
    bitfields: Vec<RedmodNamedType>,
    #[serde(default)]
    aliases: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RedmodClass {
    name: String,
    #[serde(default)]
    parent: String,
    flags: u64,
    size: usize,
    alignment: usize,
    #[serde(default)]
    properties: Vec<RedmodProperty>,
}

#[derive(Debug, Deserialize)]
struct RedmodProperty {
    name: String,
    #[serde(rename = "type")]
    red_type: String,
    offset: usize,
    flags: u64,
}

#[derive(Debug, Deserialize)]
struct RedmodNamedType {
    name: String,
}

impl From<RedmodMetadata> for RedSchema {
    fn from(metadata: RedmodMetadata) -> Self {
        let classes = metadata
            .classes
            .into_iter()
            .map(|class| {
                let name = class.name;
                let properties = class
                    .properties
                    .into_iter()
                    .map(|property| {
                        let red_type = property.red_type;
                        (
                            property.name,
                            RedProperty {
                                cs_type: red_type.clone(),
                                ordinal: None,
                                red_type: Some(red_type),
                                offset: Some(property.offset),
                                flags: Some(property.flags),
                            },
                        )
                    })
                    .collect();
                (
                    name,
                    RedClass {
                        base: (!class.parent.is_empty()).then_some(class.parent),
                        properties,
                        flags: Some(class.flags),
                        size: Some(class.size),
                        alignment: Some(class.alignment),
                    },
                )
            })
            .collect();
        Self {
            classes,
            enums: metadata.enums.into_iter().map(|value| value.name).collect(),
            bitfields: metadata
                .bitfields
                .into_iter()
                .map(|value| value.name)
                .collect(),
            aliases: metadata.aliases,
            simple_types: metadata
                .simple_types
                .into_iter()
                .map(|value| value.name)
                .collect(),
        }
    }
}

/// Generates a versioned schema from installed `REDmod` metadata and the original
/// game script bundle. Source fingerprints are stored with the schema.
///
/// # Errors
/// Returns an error if either game input is absent, malformed or cannot be read.
pub fn generate(game_root: &Path, output: &Path) -> Result<usize, SchemaError> {
    bundle::generate(game_root, output)
}

#[cfg(test)]
mod tests {
    use super::RedSchema;

    #[test]
    fn bundled_retains_every_established_class_property_type_and_order() {
        let mut expected =
            RedSchema::from_slice(include_bytes!("../data/serialization.json")).unwrap();
        let game = RedSchema::from_slice(include_bytes!("../data/schema.json")).unwrap();
        expected.merge(RedSchema {
            classes: game
                .classes
                .into_iter()
                .filter(|(_, class)| class.flags.is_some())
                .collect(),
            ..RedSchema::default()
        });
        let actual = RedSchema::bundled().unwrap();
        for (name, expected_class) in expected.classes {
            let actual_class = actual
                .classes
                .get(&name)
                .unwrap_or_else(|| panic!("missing class {name}"));
            assert_eq!(actual_class.base, expected_class.base, "{name} base");
            for (field, expected_property) in expected_class.properties {
                let actual_property = actual_class
                    .properties
                    .get(&field)
                    .unwrap_or_else(|| panic!("missing {name}.{field}"));
                assert_eq!(
                    serde_json::to_value(actual_property).unwrap(),
                    serde_json::to_value(expected_property).unwrap(),
                    "{name}.{field}"
                );
            }
        }
    }

    #[test]
    fn bundled_schema_contains_native_and_script_reflection() {
        let schema = RedSchema::bundled().unwrap();
        assert!(schema.classes.contains_key("scnSceneResource"));
        assert!(schema.classes.contains_key("ABaseQuestObjectiveWrapper"));
        assert!(
            schema.classes["AIBaseUseWorkspotCommand"]
                .properties
                .contains_key("movementType")
        );
        let provenance = RedSchema::bundled_provenance().unwrap();
        assert!(
            provenance
                .sources
                .iter()
                .any(|s| s.path == "r6/cache/final.redscripts")
        );
        assert!(
            provenance
                .sources
                .iter()
                .any(|s| s.path == "tools/redmod/metadata.json")
        );
        assert!(
            provenance
                .sources
                .iter()
                .all(|s| !s.path.contains("WolvenKit") && s.sha1.len() == 40)
        );
    }

    #[test]
    fn automatic_loader_needs_no_external_schema_and_applies_explicit_override_last() {
        let root = tempfile::tempdir().unwrap();
        let automatic = RedSchema::load(Some(root.path()), None).unwrap();
        assert!(automatic.classes.contains_key("scnSceneResource"));
        let path = root.path().join("override.json");
        std::fs::write(&path, br#"{"CustomClass":{"base":"ISerializable","properties":{"value":{"cs_type":"CInt32","ordinal":0}}}}"#).unwrap();
        let overridden = RedSchema::load(Some(root.path()), Some(&path)).unwrap();
        assert!(overridden.classes.contains_key("CustomClass"));
        assert!(overridden.classes.contains_key("scnSceneResource"));
        std::fs::write(&path, b"not json").unwrap();
        assert!(RedSchema::load(None, Some(&path)).is_err());
    }

    #[test]
    fn rejects_unknown_bundle_versions_and_invalid_script_bundles() {
        assert!(RedSchema::from_slice(br#"{"format":"ghostline-red-schema","schema_version":99,"provenance":{"generator":"test","sources":[]},"schema":{}}"#).is_err());
        assert!(super::scripts::read(b"not a script bundle").is_err());
    }

    #[test]
    fn loads_legacy_class_map() {
        let schema = RedSchema::from_slice(
            br#"{
                "Derived": {
                    "base": "Base",
                    "properties": {
                        "value": {"cs_type": "CInt32", "ordinal": 1}
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(schema.classes["Derived"].base.as_deref(), Some("Base"));
        assert_eq!(
            schema.classes["Derived"].properties["value"]
                .red_type
                .as_deref(),
            None
        );
        assert!(schema.enums.is_empty());
    }

    #[test]
    fn loads_official_redmod_metadata() {
        let schema = RedSchema::from_slice(
            br#"{
                "simple_types": [{"name": "Int32"}],
                "classes": [{
                    "name": "Derived",
                    "parent": "Base",
                    "flags": 2,
                    "size": 16,
                    "alignment": 8,
                    "properties": [{
                        "name": "mode",
                        "type": "ExampleMode",
                        "offset": 4,
                        "flags": 17
                    }],
                    "functions": [],
                    "static_functions": []
                }],
                "enums": [{
                    "name": "ExampleMode",
                    "size": 4,
                    "options": {"First": 0}
                }],
                "bitfields": [{
                    "name": "ExampleFlags",
                    "size": 4,
                    "alignment": 1,
                    "bits": {"Visible": 0}
                }],
                "aliases": {"OldDerived": "Derived"},
                "global_functions": []
            }"#,
        )
        .unwrap();

        let property = &schema.classes["Derived"].properties["mode"];
        assert_eq!(property.red_type.as_deref(), Some("ExampleMode"));
        assert_eq!(property.offset, Some(4));
        assert_eq!(property.flags, Some(17));
        assert!(schema.enums.contains("ExampleMode"));
        assert!(schema.bitfields.contains("ExampleFlags"));
        assert_eq!(schema.aliases["OldDerived"], "Derived");
        assert!(schema.class_names().contains("OldDerived"));
    }

    #[test]
    fn official_overlay_preserves_legacy_only_classes_and_properties() {
        let mut schema = RedSchema::from_slice(
            br#"{
                "LegacyOnly": {"base": null, "properties": {}},
                "Shared": {
                    "base": "LegacyBase",
                    "properties": {
                        "legacyProperty": {"cs_type": "CInt32", "ordinal": 1},
                        "mode": {"cs_type": "LegacyMode", "ordinal": 2}
                    }
                }
            }"#,
        )
        .unwrap();
        let official = RedSchema::from_slice(
            br#"{
                "simple_types": [],
                "classes": [{
                    "name": "Shared",
                    "parent": "OfficialBase",
                    "flags": 2,
                    "size": 16,
                    "alignment": 8,
                    "properties": [{
                        "name": "mode",
                        "type": "OfficialMode",
                        "offset": 4,
                        "flags": 17
                    }]
                }],
                "enums": [{"name": "OfficialMode"}],
                "bitfields": [],
                "aliases": {}
            }"#,
        )
        .unwrap();

        schema.merge(official);

        assert!(schema.classes.contains_key("LegacyOnly"));
        assert_eq!(
            schema.classes["Shared"].base.as_deref(),
            Some("OfficialBase")
        );
        assert!(
            schema.classes["Shared"]
                .properties
                .contains_key("legacyProperty")
        );
        assert_eq!(
            schema.classes["Shared"].properties["mode"]
                .red_type
                .as_deref(),
            Some("OfficialMode")
        );
    }
}
