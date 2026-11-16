use super::{AssetError, AssetReader, ResourceKey, invalid, name, root, unwrap};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Native `C2dArray` headers and rows, including empty cells.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Table2d {
    /// Native column names.
    pub headers: Vec<String>,
    /// Native row values in column order.
    pub rows: Vec<Vec<String>>,
}

/// One authored entity or appearance-resource entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Appearance {
    /// Exact name used to select this entry.
    pub name: String,
    /// Entity entry's referenced appearance resource, when present.
    pub resource: Option<ResourceKey>,
    /// Entity entry's exact appearance name inside that resource.
    pub resource_appearance: String,
}

impl AssetReader<'_> {
    /// Loads a native two-dimensional table such as a factory registration table.
    ///
    /// # Errors
    /// Returns [`AssetError`] for missing resources or malformed table data.
    pub fn table(&mut self, key: &ResourceKey) -> Result<Table2d, AssetError> {
        let document = self.document(key)?;
        let r = root(&document);
        if r.get("$type").and_then(Value::as_str) != Some("C2dArray") {
            return Err(invalid("expected C2dArray"));
        }
        let strings = |v: &Value| -> Result<Vec<String>, AssetError> {
            v.as_array()
                .ok_or_else(|| invalid("expected table array"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("expected table string"))
                })
                .collect()
        };
        let headers = strings(
            r.get("headers")
                .ok_or_else(|| invalid("missing table headers"))?,
        )?;
        let rows = match r.get("data") {
            None => Vec::new(),
            Some(data) => data
                .as_array()
                .ok_or_else(|| invalid("expected table rows"))?
                .iter()
                .map(strings)
                .collect::<Result<Vec<_>, _>>()?,
        };
        Ok(Table2d { headers, rows })
    }

    /// Lists authored appearances without guessing suffixes or resolving assemblies.
    ///
    /// # Errors
    /// Returns [`AssetError`] for missing resources, wrong root kinds or bad references.
    pub fn appearances(&mut self, key: &ResourceKey) -> Result<Vec<Appearance>, AssetError> {
        let document = self.document(key)?;
        let r = root(&document);
        if !matches!(
            r.get("$type").and_then(Value::as_str),
            Some("entEntityTemplate" | "appearanceAppearanceResource")
        ) {
            return Err(invalid("expected entity or appearance resource"));
        }
        let mut out = Vec::new();
        if let Some(entries) = r.get("appearances").and_then(Value::as_array) {
            for entry in entries {
                let entry = unwrap(entry);
                out.push(Appearance {
                    name: name(entry, "name"),
                    resource: entry
                        .get("appearanceResource")
                        .map(ResourceKey::from_value)
                        .transpose()?
                        .flatten(),
                    resource_appearance: name(entry, "appearanceName"),
                });
            }
        }
        Ok(out)
    }
}
