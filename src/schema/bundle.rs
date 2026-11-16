//! Versioned game-derived schema and reproducible source fingerprints.
use super::{RedSchema, SchemaError};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::{fs, path::Path};

/// A fingerprinted game file used to generate the bundled reflection metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Game-relative source path; never a developer's absolute path.
    pub path: String,
    /// SHA-1 fingerprint of the exact source bytes.
    pub sha1: String,
}

/// Provenance of the game files used to generate a schema snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Generator package and version.
    pub generator: String,
    /// Inputs used to produce the snapshot.
    pub sources: Vec<Source>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Document {
    pub format: String,
    pub schema_version: u32,
    pub provenance: Provenance,
    pub schema: RedSchema,
}

impl Document {
    pub fn validate(self) -> Result<Self, SchemaError> {
        if self.format != "ghostline-red-schema" || self.schema_version != 1 {
            return Err(SchemaError::Invalid(
                "unsupported bundled schema format/version".into(),
            ));
        }
        Ok(self)
    }
}

pub(super) fn generate(game: &Path, output: &Path) -> Result<usize, SchemaError> {
    let mut sources = Vec::new();
    let mut read = |path: &str| -> Result<Vec<u8>, SchemaError> {
        let bytes = fs::read(game.join(path))?;
        sources.push(Source {
            path: path.into(),
            sha1: format!("{:x}", Sha1::digest(&bytes)),
        });
        Ok(bytes)
    };
    let script_bytes = read("r6/cache/final.redscripts")?;
    let mut schema = super::scripts::read(&script_bytes)?;
    schema.merge(RedSchema::from_slice(&read("tools/redmod/metadata.json")?)?);
    if game.join("bin/x64/Cyberpunk2077.exe").is_file() {
        read("bin/x64/Cyberpunk2077.exe")?;
    }
    let count = schema.classes.len();
    let document = Document {
        format: "ghostline-red-schema".into(),
        schema_version: 1,
        provenance: Provenance {
            generator: format!("ghostline-red {}", env!("CARGO_PKG_VERSION")),
            sources,
        },
        schema,
    };
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec(&document)?)?;
    Ok(count)
}
