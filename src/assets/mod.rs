//! Typed native resources used to assemble and animate visual entities.

pub(crate) mod assembly;
mod collision;
mod customization;
mod mesh_binding;
mod rig;
mod table;

pub use assembly::{Assembly, AssemblyComponent, Binding, Slot};
pub use customization::{Customization, CustomizationControl};
pub use mesh_binding::MeshAppearanceBinding;
pub use rig::{Bone, Rig};
pub use table::{Appearance, Table2d};

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

use glam::{Mat4, Quat, Vec3};
use serde_json::Value;

use crate::{codec, material::ArchiveSet, schema::RedSchema};

/// Failure to resolve or decode a native visual resource.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// Archive lookup or decompression failed.
    #[error(transparent)]
    Archive(#[from] crate::material::MaterialError),
    /// Reflected native decoding failed.
    #[error(transparent)]
    Codec(#[from] codec::CodecError),
    /// Embedded binary data is malformed.
    #[error(transparent)]
    Base64(#[from] base64::DecodeError),
    /// A resource has invalid or unsupported visual data.
    #[error("invalid visual resource: {0}")]
    Invalid(String),
}

/// Exact resource identity, including references with no known depot spelling.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ResourceKey {
    /// Depot path, with native separators.
    Path(String),
    /// Native 64-bit depot hash.
    Hash(#[serde(with = "decimal_hash")] u64),
}

mod decimal_hash {
    use serde::{Deserialize as _, Deserializer, Serializer, de::Error as _};

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "Serde's with callback receives a reference to the field"
    )]
    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}

impl ResourceKey {
    /// Native depot hash, independent of whether the path spelling is known.
    #[must_use]
    pub fn native_hash(&self) -> u64 {
        match self {
            Self::Path(path) => crate::archive::depot_path_hash(path),
            Self::Hash(hash) => *hash,
        }
    }
    /// Parses a depot spelling or decimal native hash.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        value
            .parse()
            .map_or_else(|_| Self::Path(value.replace('/', "\\")), Self::Hash)
    }

    /// Reads a reflected resource-reference property. Empty references return `None`.
    ///
    /// # Errors
    /// Returns [`AssetError`] for an invalid reference representation.
    pub fn from_value(value: &Value) -> Result<Option<Self>, AssetError> {
        let path = value.get("DepotPath").unwrap_or(value);
        let Some(text) = text(path) else {
            if let Some(hash) = path.as_u64() {
                return Ok((hash != 0).then_some(Self::Hash(hash)));
            }
            return if path.is_null() {
                Ok(None)
            } else {
                Err(invalid("resource reference has no depot path/hash"))
            };
        };
        Ok((!text.is_empty() && text != "0" && text != "None").then(|| Self::parse(text)))
    }
}

impl fmt::Display for ResourceKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(path) => formatter.write_str(path),
            Self::Hash(hash) => hash.fmt(formatter),
        }
    }
}

/// Local translation, XYZW quaternion and scale in native RED coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    /// Translation in metres, Z up.
    pub translation: [f32; 3],
    /// Quaternion components X, Y, Z, W.
    pub rotation: [f32; 4],
    /// Axis scale.
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

impl Transform {
    /// Forms a native local-to-parent matrix after validating the transform.
    ///
    /// # Errors
    /// Returns [`AssetError`] for nonfinite or singular scale data. The authored
    /// all-zero quaternion has no rotation and is evaluated as identity.
    pub fn matrix(self) -> Result<Mat4, AssetError> {
        let translation = Vec3::from(self.translation);
        let scale = Vec3::from(self.scale);
        let rotation = Quat::from_array(self.rotation);
        if !translation.is_finite()
            || !scale.is_finite()
            || !rotation.is_finite()
            || scale.cmpeq(Vec3::ZERO).any()
        {
            return Err(invalid("nonfinite or singular transform"));
        }
        Ok(Mat4::from_scale_rotation_translation(
            scale,
            if rotation.length_squared() == 0.0 {
                Quat::IDENTITY
            } else {
                rotation.normalize()
            },
            translation,
        ))
    }

    pub(crate) fn decode(value: &Value) -> Result<Self, AssetError> {
        let translation = vector(
            value.get("Translation").or_else(|| value.get("Position")),
            [0.0; 3],
        )?;
        let scale = vector(value.get("Scale"), [1.0; 3])?;
        let mut rotation = [0.0, 0.0, 0.0, 1.0];
        if let Some(source) = value.get("Rotation").or_else(|| value.get("Orientation")) {
            for (index, names) in [["i", "X"], ["j", "Y"], ["k", "Z"], ["r", "W"]]
                .iter()
                .enumerate()
            {
                if let Some(component) = source.get(names[0]).or_else(|| source.get(names[1])) {
                    rotation[index] = number(component)?;
                }
            }
        }
        let result = Self {
            translation,
            rotation,
            scale,
        };
        result.matrix()?;
        Ok(result)
    }
}

/// Cached reflected-resource access over an existing archive index and schema.
#[derive(Debug)]
pub struct AssetReader<'a> {
    archives: &'a ArchiveSet,
    schema: &'a RedSchema,
    documents: HashMap<ResourceKey, Arc<Value>>,
    shared: Option<&'a DocumentCache>,
}

/// Bounded reflected documents shared by readers of one archive/schema snapshot.
/// Do not reuse this cache after changing either source snapshot.
#[derive(Debug)]
pub struct DocumentCache {
    collision: Mutex<Option<Arc<crate::physics::GeometryCache>>>,
    collision_names: Mutex<Option<Arc<HashMap<u64, String>>>>,
    documents: Mutex<HashMap<ResourceKey, Arc<Value>>>,
    locks: [Mutex<()>; 64],
}

impl Default for DocumentCache {
    fn default() -> Self {
        Self {
            documents: Mutex::new(HashMap::new()),
            collision: Mutex::new(None),
            collision_names: Mutex::new(None),
            locks: std::array::from_fn(|_| Mutex::new(())),
        }
    }
}

impl<'a> AssetReader<'a> {
    /// Reads newline-separated depot name hints, optionally KARK compressed.
    /// Names do not establish that a resource exists in the selected archives.
    ///
    /// # Errors
    /// Returns an error for invalid compression or non-UTF-8 name data.
    pub fn resource_names(bytes: &[u8]) -> Result<Vec<String>, AssetError> {
        let decoded;
        let bytes = if bytes.starts_with(b"KARK") {
            let size = bytes
                .get(4..8)
                .ok_or_else(|| invalid("truncated resource name dictionary"))?;
            decoded = crate::archive::decompress_payload(
                &bytes[8..],
                u32::from_le_bytes(size.try_into().expect("four-byte slice")) as usize,
            )
            .map_err(crate::material::MaterialError::from)?;
            decoded.as_slice()
        } else {
            bytes
        };
        let names = std::str::from_utf8(bytes)
            .map_err(|error| invalid(format!("resource names are not UTF-8: {error}")))?;
        Ok(names
            .trim_start_matches('\u{feff}')
            .lines()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(|name| name.replace('/', "\\"))
            .collect())
    }
    /// Reads the original native resource bytes, preserving its serialization tables.
    ///
    /// # Errors
    /// Returns an archive error when the resource cannot be read.
    pub fn read_bytes(&self, key: &ResourceKey) -> Result<Vec<u8>, AssetError> {
        Ok(self.archives.read_resource_hash(key.native_hash())?)
    }

    /// Checks whether the resource exists in the indexed native archives.
    pub fn resource_exists(&self, key: &ResourceKey) -> bool {
        self.archives.contains_resource_hash(key.native_hash())
    }
    /// Creates a reader without rebuilding the archive index.
    #[must_use]
    pub fn new(archives: &'a ArchiveSet, schema: &'a RedSchema) -> Self {
        Self {
            archives,
            schema,
            documents: HashMap::new(),
            shared: None,
        }
    }

    /// Shares decoded documents across readers of the same archive/schema snapshot.
    #[must_use]
    pub fn with_cache(
        archives: &'a ArchiveSet,
        schema: &'a RedSchema,
        cache: &'a DocumentCache,
    ) -> Self {
        Self {
            shared: Some(cache),
            ..Self::new(archives, schema)
        }
    }

    /// Decodes a native resource once and shares its reflected document.
    ///
    /// # Errors
    /// Returns [`AssetError`] for absent resources or invalid native data.
    pub fn document(&mut self, key: &ResourceKey) -> Result<Arc<Value>, AssetError> {
        if let Some(document) = self.documents.get(key) {
            return Ok(Arc::clone(document));
        }
        let _guard = self.shared.map(|cache| {
            cache.locks[(key.native_hash() & 63) as usize]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        if let Some(cache) = self.shared
            && let Some(document) = cache
                .documents
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(key)
                .cloned()
        {
            self.documents.insert(key.clone(), Arc::clone(&document));
            return Ok(document);
        }
        let bytes = match key {
            ResourceKey::Path(path) => self.archives.read_resource(path)?,
            ResourceKey::Hash(hash) => self.archives.read_resource_hash(*hash)?,
        };
        let document = Arc::new(codec::decode_wkit_from_bytes(&bytes, self.schema)?);
        if let Some(cache) = self.shared {
            let mut documents = cache
                .documents
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if documents.len() >= 64
                && let Some(old) = documents.keys().next().cloned()
            {
                documents.remove(&old);
            }
            documents.insert(key.clone(), Arc::clone(&document));
        }
        self.documents.insert(key.clone(), Arc::clone(&document));
        Ok(document)
    }

    /// Decodes a rig's named bones, hierarchy and local rest transforms.
    ///
    /// # Errors
    /// Returns [`AssetError`] for invalid resources, transforms or hierarchies.
    pub fn rig(&mut self, key: &ResourceKey) -> Result<Rig, AssetError> {
        let document = self.document(key)?;
        let mut rig = Rig::decode(&document)?;
        rig.source = Some(key.clone());
        Ok(rig)
    }

    /// Decodes a morphtarget's embedded shape, sparse targets and base-mesh palette.
    ///
    /// # Errors
    /// Returns [`AssetError`] for missing sources or malformed native morph data.
    pub fn morph_mesh(&mut self, key: &ResourceKey) -> Result<crate::morph::MorphMesh, AssetError> {
        let document = self.document(key)?;
        let base = crate::morph::base_mesh(&document)?;
        let base = self.document(&base)?;
        crate::morph::decode(&document, &base)
    }
}

/// Compares decoded native JSON values, retaining exact integers and f32
/// semantics for floating-point fields across JSON serialization round trips.
pub fn same_native_value(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "native RED floating fields are f32"
            )]
            {
                a.as_f64().map(|v| v as f32) == b.as_f64().map(|v| v as f32)
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_native_value(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| same_native_value(a, b)))
        }
        _ => left == right,
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> AssetError {
    AssetError::Invalid(message.into())
}
pub(crate) fn root(document: &Value) -> &Value {
    document.pointer("/Data/RootChunk").unwrap_or(document)
}
pub(crate) fn unwrap(value: &Value) -> &Value {
    value.get("Data").unwrap_or(value)
}
pub(crate) fn text(value: &Value) -> Option<&str> {
    value.get("$value").unwrap_or(value).as_str()
}
pub(crate) fn name(value: &Value, field: &str) -> String {
    value
        .get(field)
        .and_then(text)
        .unwrap_or_default()
        .to_owned()
}
pub(crate) fn number(value: &Value) -> Result<f32, AssetError> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "native transforms use finite f32 components"
    )]
    let result = value
        .as_f64()
        .ok_or_else(|| invalid("expected numeric transform component"))? as f32;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(invalid("nonfinite transform component"))
    }
}
pub(crate) fn vector(
    value: Option<&Value>,
    mut defaults: [f32; 3],
) -> Result<[f32; 3], AssetError> {
    if let Some(value) = value {
        for (index, axis) in ["X", "Y", "Z"].iter().enumerate() {
            if let Some(component) = value
                .get(axis)
                .or_else(|| value.get(axis.to_ascii_lowercase()))
            {
                defaults[index] =
                    if component.get("$type").and_then(Value::as_str) == Some("FixedPoint") {
                        let bits = component
                            .get("Bits")
                            .and_then(|v| v.as_i64().or_else(|| v.as_str()?.parse().ok()))
                            .and_then(|v| i32::try_from(v).ok())
                            .ok_or_else(|| {
                                invalid("FixedPoint.Bits must be a signed 32-bit integer")
                            })?;
                        bits as f32 / 131072.
                    } else {
                        number(component)?
                    };
            }
        }
    }
    Ok(defaults)
}

#[cfg(test)]
mod native_value_tests {
    use super::same_native_value;
    use serde_json::json;

    #[test]
    fn float_round_trips_do_not_change_bindings_or_relax_integer_identity() {
        assert!(same_native_value(
            &json!([0.09999847412109376]),
            &json!([0.09999847412109375])
        ));
        assert!(!same_native_value(&json!([0.09999847]), &json!([0.1001])));
        assert!(!same_native_value(
            &json!(18_446_744_073_709_551_614_u64),
            &json!(18_446_744_073_709_551_615_u64)
        ));
    }
}
