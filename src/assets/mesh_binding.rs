//! Native mesh appearance and material-slot bindings without texture preparation.
#[cfg(test)]
use super::name;
use super::{AssetError, AssetReader, ResourceKey, invalid, root, unwrap};

/// Effective native component mesh appearance and its material-source resource.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MeshAppearanceBinding {
    /// Selected mesh or morph geometry.
    pub geometry: ResourceKey,
    /// Mesh supplying the material definitions, including for morph geometry.
    pub material_source: ResourceKey,
    /// Exact authored mesh appearance.
    pub appearance: String,
    /// Component request before applying the native first-appearance fallback.
    pub requested_appearance: String,
    /// Whether the request resolved through the native first-appearance rule.
    pub used_native_default: bool,
    /// Authored material names in native render-chunk order.
    pub chunk_materials: Vec<String>,
    /// Native geometry is exclusively a shadow rendering resource.
    pub shadow_only: bool,
    /// The selected mesh has no authored render geometry.
    pub non_visual: bool,
}

impl AssetReader<'_> {
    /// Lists exact material appearance names, including native omitted-name defaults.
    /// Morph geometry uses the appearances of its declared base mesh.
    ///
    /// # Errors
    /// Returns native resource/type errors or an error for an empty appearance table.
    pub fn mesh_appearance_names(
        &mut self,
        geometry: &ResourceKey,
    ) -> Result<Vec<String>, AssetError> {
        let document = self.document(geometry)?;
        let material_source = if root(&document)["$type"] == "MorphTargetMesh" {
            crate::morph::base_mesh(&document).map_err(|error| invalid(error.to_string()))?
        } else {
            geometry.clone()
        };
        let document = self.document(&material_source)?;
        let mesh = root(&document);
        if !matches!(
            mesh["$type"].as_str(),
            Some("CMesh" | "CPhysicsDecorationResource")
        ) {
            return Err(invalid(format!("{material_source} is not a mesh")));
        }
        let entries = mesh["appearances"]
            .as_array()
            .filter(|entries| !entries.is_empty())
            .ok_or_else(|| invalid("mesh has no appearances"))?;
        Ok(entries.iter().map(unwrap).map(appearance_name).collect())
    }
    /// Resolves material slots using native exact-match/first-appearance semantics.
    /// No textures are extracted and no GPU is allocated. Root entity/appearance
    /// selection remains exact; this fallback applies to component mesh appearances.
    ///
    /// # Errors
    /// Returns [`AssetError`] for missing resources or meshes with no authored appearances.
    pub fn mesh_appearance_binding(
        &mut self,
        geometry: &ResourceKey,
        appearance: &str,
    ) -> Result<MeshAppearanceBinding, AssetError> {
        let document = self.document(geometry)?;
        let material_source = if root(&document)["$type"] == "MorphTargetMesh" {
            crate::morph::base_mesh(&document).map_err(|e| invalid(e.to_string()))?
        } else {
            geometry.clone()
        };
        let document = self.document(&material_source)?;
        let mesh = root(&document);
        if !matches!(
            mesh["$type"].as_str(),
            Some("CMesh" | "CPhysicsDecorationResource")
        ) {
            return Err(invalid(format!("{material_source} is not a mesh")));
        }
        let appearances = mesh["appearances"]
            .as_array()
            .ok_or_else(|| invalid("mesh has no appearances"))?;
        let selected = select_appearance(appearances, appearance)?;
        let effective = appearance_name(selected);
        let non_visual = mesh
            .get("renderResourceBlob")
            .is_none_or(|value| value.is_null());
        let empty = Vec::new();
        let unassigned = mesh
            .get("materialEntries")
            .and_then(serde_json::Value::as_array)
            .is_none_or(Vec::is_empty);
        let materials = selected["chunkMaterials"]
            .as_array()
            .or_else(|| (non_visual || unassigned).then_some(&empty))
            .ok_or_else(|| invalid("mesh appearance has no material slots"))?;
        let chunk_materials = materials
            .iter()
            .map(|value| {
                super::text(value)
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("invalid chunk material name"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MeshAppearanceBinding {
            non_visual,
            geometry: geometry.clone(),
            material_source,
            used_native_default: effective != appearance,
            appearance: effective,
            requested_appearance: appearance.into(),
            chunk_materials,
            shadow_only: ["isShadowMesh", "isPlayerShadowMesh"]
                .iter()
                .any(|key| mesh[*key].as_bool().unwrap_or(false)),
        })
    }
}

fn select_appearance<'a>(
    appearances: &'a [serde_json::Value],
    requested: &str,
) -> Result<&'a serde_json::Value, AssetError> {
    appearances
        .iter()
        .map(unwrap)
        .find(|entry| appearance_name(entry) == requested)
        .or_else(|| appearances.first().map(unwrap))
        .ok_or_else(|| invalid("mesh has no authored appearances"))
}

fn appearance_name(entry: &serde_json::Value) -> String {
    entry
        .get("name")
        .and_then(super::text)
        .unwrap_or("default")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn component_defaults_follow_native_array_order_and_exact_requests_win() {
        let appearances = vec![
            json!({"Data":{"name":"zinc"}}),
            json!({"Data":{"name":"amber"}}),
        ];
        assert_eq!(
            name(select_appearance(&appearances, "default").unwrap(), "name"),
            "zinc"
        );
        assert_eq!(
            name(select_appearance(&appearances, "amber").unwrap(), "name"),
            "amber"
        );
        assert_eq!(
            name(select_appearance(&appearances, "missing").unwrap(), "name"),
            "zinc"
        );
        assert!(select_appearance(&[], "default").is_err());
        let unnamed = vec![json!({"Data":{"chunkMaterials":["shadow"]}})];
        assert_eq!(
            appearance_name(select_appearance(&unnamed, "default").unwrap()),
            "default"
        );
    }
}
