//! Native compiled `TweakDB` tables and exact record/property identities.
//!
//! Version 8/parser 4 caches contain shared typed value pools, flat-to-value
//! mappings, record class hashes, queries and group tags. No name dictionary is
//! required to address a record property. Optional names are lookup hints only.

mod reader;

use crate::schema::RedSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

/// Malformed, truncated or unsupported native cache data.
#[derive(Debug, thiserror::Error)]
pub enum TweakError {
    /// A native table or value is malformed.
    #[error("invalid TweakDB at byte {offset}: {reason}")]
    Invalid { offset: usize, reason: String },
    /// The version requires a different reader.
    #[error("unsupported TweakDB version {blob}/{parser}")]
    Version { blob: u32, parser: u32 },
    /// The type pool cannot be decoded safely.
    #[error("unsupported TweakDB flat type hash {0}")]
    Type(u64),
    /// Name dictionary decompression failed.
    #[error(transparent)]
    Compression(#[from] crate::kraken::KrakenError),
}

/// Native CRC32/UTF-16-length identity, serialized as a decimal string.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TweakId(pub u64);

impl TweakId {
    /// Hashes a native name using its UTF-8 CRC and UTF-16 code-unit length.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        Self(
            u64::from(crc32fast::hash(name.as_bytes()))
                | ((name.encode_utf16().count() as u64) << 32),
        )
    }

    /// Derives a flat identity without needing the original record name.
    #[must_use]
    pub fn property(self, name: &str) -> Self {
        let [a, b, c, d, _, _, _, _] = self.0.to_le_bytes();
        let mut hash = crc32fast::Hasher::new_with_initial(u32::from_le_bytes([a, b, c, d]));
        hash.update(b".");
        hash.update(name.as_bytes());
        Self(
            u64::from(hash.finalize())
                | (((self.0 >> 32) + 1 + name.encode_utf16().count() as u64) << 32),
        )
    }
}

impl std::fmt::Display for TweakId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for TweakId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0.to_string())
    }
}
impl<'de> Deserialize<'de> for TweakId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        String::deserialize(d)?
            .parse()
            .map(Self)
            .map_err(D::Error::custom)
    }
}

/// Typed flat payload. Native 64-bit integers and identities use decimal strings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatValue {
    /// RED type spelling, including the `array:` prefix where applicable.
    pub native_type: String,
    /// Decoded scalar, string, vector/object or array.
    pub value: Value,
}

/// A decoded compiled cache. Values shared by many flats are stored once.
#[derive(Debug)]
pub struct TweakDb {
    /// Native record identity to concrete record class hash.
    pub records: BTreeMap<TweakId, u32>,
    /// Flat identities to shared typed values.
    pub flats: HashMap<TweakId, Arc<FlatValue>>,
    /// Native query identities and ordered result lists.
    pub queries: BTreeMap<TweakId, Vec<TweakId>>,
    /// Native per-record group flags.
    pub group_tags: BTreeMap<TweakId, u8>,
    /// Header record checksum, retained without interpreting it as content identity.
    pub record_checksum: u32,
}

/// Names verified against an existing cache and its flat identities.
#[derive(Debug, Default)]
pub struct TweakNames {
    /// Known native record names, including names recovered from flat prefixes.
    pub records: HashMap<TweakId, String>,
    /// Known flat property names grouped by owning native record.
    pub properties: HashMap<TweakId, BTreeMap<String, TweakId>>,
}

impl TweakNames {
    /// Matches dictionary strings to cache identities; unmatched hints are ignored.
    #[must_use]
    pub fn resolve(db: &TweakDb, names: impl IntoIterator<Item = String>) -> Self {
        let mut out = Self::default();
        for name in names {
            let id = TweakId::from_name(&name);
            if db.records.contains_key(&id) {
                out.records.insert(id, name.clone());
            }
            if db.flats.contains_key(&id)
                && let Some((owner, property)) = name.rsplit_once('.')
            {
                let record = TweakId::from_name(owner);
                if db.records.contains_key(&record) {
                    out.records
                        .entry(record)
                        .or_insert_with(|| owner.to_owned());
                    out.properties
                        .entry(record)
                        .or_default()
                        .insert(property.to_owned(), id);
                }
            }
        }
        out
    }
}

impl TweakDb {
    /// Decodes a complete native cache, validating pool indices and table bounds.
    ///
    /// # Errors
    /// Returns [`TweakError`] for unsupported or malformed data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TweakError> {
        reader::decode(bytes)
    }

    /// Looks up an exact record property, including unnamed records.
    #[must_use]
    pub fn property(&self, id: TweakId, name: &str) -> Option<&FlatValue> {
        self.flats.get(&id.property(name)).map(AsRef::as_ref)
    }
}

/// Record class hash resolution and inheritance from RED reflection metadata.
#[derive(Debug)]
pub struct RecordTypes {
    names: HashMap<u32, String>,
    parents: BTreeMap<String, Option<String>>,
}

impl RecordTypes {
    /// Indexes reflected record classes; unknown native hashes remain unresolved.
    #[must_use]
    pub fn new(schema: &RedSchema) -> Self {
        let mut names = HashMap::new();
        for name in schema.classes.keys() {
            if let Some(short) = name
                .strip_prefix("gamedata")
                .and_then(|s| s.strip_suffix("_Record"))
            {
                names.insert(record_type_hash(short), name.clone());
            }
        }
        Self {
            names,
            parents: schema
                .classes
                .iter()
                .map(|(n, c)| (n.clone(), c.base.clone()))
                .collect(),
        }
    }

    /// Resolves a native record class hash.
    #[must_use]
    pub fn name(&self, hash: u32) -> Option<&str> {
        self.names.get(&hash).map(String::as_str)
    }

    /// Tests class ancestry, terminating safely on malformed reflection cycles.
    #[must_use]
    pub fn is_a(&self, hash: u32, ancestor: &str) -> bool {
        let mut name = self.name(hash);
        let mut seen = HashSet::new();
        while let Some(current) = name {
            if current == ancestor {
                return true;
            }
            if !seen.insert(current) {
                break;
            }
            name = self.parents.get(current).and_then(|p| p.as_deref());
        }
        false
    }
}

/// Murmur3 x86-32 record-class hash, using the engine's `0x5EEDBA5E` seed.
#[must_use]
pub fn record_type_hash(short_name: &str) -> u32 {
    let bytes = short_name.as_bytes();
    let mut h = 0x5eed_ba5e_u32;
    for chunk in bytes.chunks_exact(4) {
        let k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        h ^= k
            .wrapping_mul(0xcc9e_2d51)
            .rotate_left(15)
            .wrapping_mul(0x1b87_3593);
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let mut tail = 0_u32;
    for (i, b) in bytes.chunks_exact(4).remainder().iter().enumerate() {
        tail |= u32::from(*b) << (i * 8);
    }
    if !bytes.len().is_multiple_of(4) {
        h ^= tail
            .wrapping_mul(0xcc9e_2d51)
            .rotate_left(15)
            .wrapping_mul(0x1b87_3593);
    }
    let length_bytes = (bytes.len() as u64).to_le_bytes();
    h ^= u32::from_le_bytes([
        length_bytes[0],
        length_bytes[1],
        length_bytes[2],
        length_bytes[3],
    ]);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Reads a length-prefixed native name table, optionally wrapped in KARK.
///
/// The decompressed dictionary has a 20-byte header. Names are hints and must
/// be matched to an existing native identity before assigning them to records.
///
/// # Errors
/// Returns [`TweakError`] for malformed strings, excessive size or bad compression.
pub fn read_names(bytes: &[u8]) -> Result<Vec<String>, TweakError> {
    reader::names(bytes)
}
