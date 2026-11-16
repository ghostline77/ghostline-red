//! Native customization controls with lossless option and activation metadata.

use super::{AssetError, AssetReader, ResourceKey, invalid, text};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One native control in its owning domain. Slots are not unique identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomizationControl {
    /// Head, body or arms resource domain.
    pub domain: String,
    /// Source position, scoped to the source resource snapshot.
    pub source_index: usize,
    /// Native named control identity, when serialized.
    pub name: Option<String>,
    /// Native concrete control class.
    pub native_class: String,
    /// Native UI slot, when serialized.
    pub ui_slot: Option<String>,
    /// Full decoded control, including omitted/default distinctions and choices.
    pub data: Value,
}

/// Native character customization graph, including groups and perspective links.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Customization {
    /// Source resource identity.
    pub source: ResourceKey,
    /// Control definitions across all domains.
    pub controls: Vec<CustomizationControl>,
    /// Root metadata excluding the control arrays, preserved without normalization.
    pub metadata: Value,
}

impl Customization {
    /// Decodes a reflected native customization root without guessing defaults.
    ///
    /// # Errors
    /// Returns [`AssetError`] for an incorrect root or malformed control arrays.
    pub fn from_document(source: ResourceKey, document: &Value) -> Result<Self, AssetError> {
        let root = document.pointer("/Data/RootChunk").unwrap_or(document);
        if root.get("$type").and_then(Value::as_str)
            != Some("gameuiCharacterCustomizationInfoResource")
        {
            return Err(invalid("expected character customization root"));
        }
        let mut metadata = root.clone();
        let mut controls = Vec::new();
        for domain in ["head", "body", "arms"] {
            let key = format!("{domain}CustomizationOptions");
            let entries = root
                .get(&key)
                .and_then(Value::as_array)
                .ok_or_else(|| invalid(format!("missing {key}")))?;
            for (index, entry) in entries.iter().enumerate() {
                let data = entry.get("Data").unwrap_or(entry);
                let class = data
                    .get("$type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("customization control has no class"))?;
                controls.push(CustomizationControl {
                    domain: domain.into(),
                    source_index: index,
                    name: data.get("name").and_then(text).map(str::to_owned),
                    native_class: class.into(),
                    ui_slot: data.get("uiSlot").and_then(text).map(str::to_owned),
                    data: data.clone(),
                });
            }
            if let Some(object) = metadata.as_object_mut() {
                object.remove(&key);
            }
        }
        Ok(Self {
            source,
            controls,
            metadata,
        })
    }
}

impl AssetReader<'_> {
    /// Loads native head, body and arm customization definitions.
    ///
    /// # Errors
    /// Returns [`AssetError`] for absent resources or malformed controls.
    pub fn customization(&mut self, key: &ResourceKey) -> Result<Customization, AssetError> {
        let document = self.document(key)?;
        Customization::from_document(key.clone(), &document)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preserves_shared_slots_nested_activation_and_baselines() {
        let root = json!({"$type":"gameuiCharacterCustomizationInfoResource",
            "headCustomizationOptions":[{"Data":{"$type":"gameuiSwitcherInfo","name":{"$value":"hair"},"uiSlot":{"$value":"hairstyle"},"options":[{"names":["color","color_fpp"]}]}}],
            "bodyCustomizationOptions":[{"Data":{"$type":"gameuiAppearanceInfo","uiSlot":{"$value":"genitals_size"}}},{"Data":{"$type":"gameuiMorphInfo","uiSlot":{"$value":"genitals_size"},"morphNames":[{"index":1}]}}],
            "armsCustomizationOptions":[],"perspectiveInfo":[{"group":"FPP"}]});
        let c = Customization::from_document(
            ResourceKey::Path("test.inkcharcustomization".into()),
            &root,
        )
        .unwrap();
        assert_eq!(c.controls.len(), 3);
        assert_eq!(c.controls[1].ui_slot, c.controls[2].ui_slot);
        assert_ne!(c.controls[1].native_class, c.controls[2].native_class);
        assert_eq!(
            c.controls[0].data["options"][0]["names"],
            json!(["color", "color_fpp"])
        );
        assert!(
            c.controls[2].data["morphNames"][0]
                .get("morphName")
                .is_none()
        );
        assert!(c.metadata.get("perspectiveInfo").is_some());
        assert!(c.metadata.get("headCustomizationOptions").is_none());
    }
}
