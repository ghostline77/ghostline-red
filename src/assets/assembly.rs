use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::Mat4;
use serde_json::Value;

use super::{
    AssetError, AssetReader, ResourceKey, Rig, Transform, invalid, name, root, text, unwrap, vector,
};

/// Component or skeleton attachment named by a native transform binding.
#[derive(Clone, Debug, Default)]
pub struct Binding {
    /// Component to which this component is bound.
    pub component: String,
    /// Bone or slot on that component; empty means its component origin.
    pub slot: String,
}

/// Named rigid attachment defined on a slot component.
#[derive(Clone, Debug)]
pub struct Slot {
    /// Name referenced by child component bindings.
    pub name: String,
    /// Rig bone to which the slot is attached.
    pub bone: String,
    /// Offset relative to the bone, in RED coordinates.
    pub transform: Transform,
}

/// A visual or transform-only component from an entity/appearance definition.
#[derive(Clone, Debug)]
pub struct AssemblyComponent {
    /// Original native component fields with self-contained transform/control
    /// bindings. Authoring tools retain flags and component-specific properties
    /// instead of reconstructing an approximate component from preview fields.
    pub template: std::sync::Arc<Value>,
    /// Native component name.
    pub name: String,
    /// Reflected native component type.
    pub kind: String,
    /// Local component transform including non-collapsed visual scale, in RED
    /// coordinates. Collapsed geometry retains its unscaled attachment frame;
    /// the original visual scale is retained in `template`.
    pub local: Mat4,
    /// Parent transform binding.
    pub parent: Binding,
    /// Control-rig inheritance binding.
    pub control: Binding,
    /// Authored rig resource.
    pub rig: Option<ResourceKey>,
    /// Mesh resource, including unnamed hashed references.
    pub mesh: Option<ResourceKey>,
    /// Effective native mesh appearance after component overrides and fallback.
    pub mesh_appearance: String,
    /// Authored component/override request; absent when the field was omitted.
    pub mesh_appearance_requested: Option<String>,
    /// Native render-chunk mask.
    pub chunk_mask: u64,
    /// Authored enable flag.
    pub enabled: bool,
    /// Authored rigid attachment slots.
    pub slots: Vec<Slot>,
}

impl AssemblyComponent {
    /// Whether native visual scale collapses this component's mesh. Its parent
    /// frame and children remain valid even when the mesh has no visible area.
    #[must_use]
    pub fn is_collapsed(&self) -> bool {
        let scale = &self.template["visualScale"];
        ["X", "Y", "Z"].iter().any(|axis| {
            scale
                .get(*axis)
                .or_else(|| scale.get(axis.to_ascii_lowercase()))
                .and_then(Value::as_f64)
                == Some(0.0)
        })
    }
}

/// Exact entity/appearance selection with expanded part entities and overrides.
#[derive(Clone, Debug)]
pub struct Assembly {
    /// Top-level requested resource.
    pub source: ResourceKey,
    /// Exact selected entity/appearance name, empty for a simple entity.
    pub appearance: String,
    /// Components in deterministic source order, including transform-only nodes.
    pub components: Vec<AssemblyComponent>,
    /// Decoded rigs referenced by the components.
    pub rigs: HashMap<ResourceKey, Rig>,
}

impl AssetReader<'_> {
    /// Resolves an exact entity or appearance selection, part entities and rigs.
    ///
    /// Component overrides are applied by native name. Randomized appearance
    /// groups require the caller to select a concrete authored appearance.
    ///
    /// # Errors
    /// Returns [`AssetError`] for missing/ambiguous selections, part cycles,
    /// unresolved handles or malformed visual data.
    pub fn assembly(
        &mut self,
        source: &ResourceKey,
        appearance: Option<&str>,
    ) -> Result<Assembly, AssetError> {
        let document = self.document(source)?;
        self.assembly_document(source, &document, appearance)
    }

    /// Resolves an authored native document using archive resources for its dependencies.
    ///
    /// # Errors
    /// Returns the same selection, binding and dependency errors as [`Self::assembly`].
    pub fn assembly_document(
        &mut self,
        source: &ResourceKey,
        document: &Value,
        appearance: Option<&str>,
    ) -> Result<Assembly, AssetError> {
        self.assembly_document_selected(
            source,
            document,
            appearance,
            None,
            &BTreeSet::new(),
            &BTreeMap::new(),
            true,
        )
    }

    /// Resolves an assembly while retaining explicitly excluded mesh components
    /// as native transform nodes without loading their mesh/material dependencies.
    ///
    /// # Errors
    /// Returns native selection, rig and non-excluded dependency errors.
    pub fn assembly_excluding_meshes(
        &mut self,
        source: &ResourceKey,
        appearance: Option<&str>,
        excluded: &std::collections::BTreeSet<String>,
    ) -> Result<Assembly, AssetError> {
        self.assembly_with_resource_overrides(source, appearance, excluded, &BTreeMap::new())
    }

    /// Resolves explicit resource corrections before loading parts, rigs and
    /// material bindings. Keys are requested native hashes; templates retain the
    /// original fields and effective component resources identify corrected data.
    ///
    /// # Errors
    /// Returns native selection, dependency or malformed-resource errors.
    pub fn assembly_with_resource_overrides(
        &mut self,
        source: &ResourceKey,
        appearance: Option<&str>,
        excluded: &std::collections::BTreeSet<String>,
        overrides: &std::collections::BTreeMap<String, ResourceKey>,
    ) -> Result<Assembly, AssetError> {
        // A patch may restore a formerly missing native reference. Prefer it;
        // corrections apply only when the requested resource is unavailable and
        // the replacement really exists in the selected source snapshot.
        let overrides = overrides
            .iter()
            .filter(|(requested, replacement)| {
                !self.resource_exists(&ResourceKey::parse(requested))
                    && self.resource_exists(replacement)
            })
            .map(|(requested, replacement)| (requested.clone(), replacement.clone()))
            .collect();
        let effective = redirected(source.clone(), &overrides);
        let document = self.document(&effective)?;
        self.assembly_document_selected(
            source, &document, appearance, None, excluded, &overrides, true,
        )
    }

    /// Resolves only explicitly selected native components and their rigs/material bindings.
    ///
    /// # Errors
    /// Returns selection/dependency errors for the requested components.
    pub fn assembly_components(
        &mut self,
        source: &ResourceKey,
        appearance: Option<&str>,
        names: &std::collections::BTreeSet<String>,
    ) -> Result<Assembly, AssetError> {
        let document = self.document(source)?;
        self.assembly_document_selected(
            source,
            &document,
            appearance,
            Some(names),
            &BTreeSet::new(),
            &BTreeMap::new(),
            true,
        )
    }

    /// Expands native entity/appearance membership without loading mesh materials
    /// or rigs. Requested mesh appearances remain unresolved. Use this for resource
    /// discovery, then resolve the discovered selections before publication.
    ///
    /// # Errors
    /// Returns native root, exact-appearance and part-expansion errors.
    pub fn assembly_manifest(
        &mut self,
        source: &ResourceKey,
        appearance: Option<&str>,
    ) -> Result<Assembly, AssetError> {
        let document = self.document(source)?;
        self.assembly_document_selected(
            source,
            &document,
            appearance,
            None,
            &BTreeSet::new(),
            &BTreeMap::new(),
            false,
        )
    }

    fn assembly_document_selected(
        &mut self,
        source: &ResourceKey,
        document: &Value,
        appearance: Option<&str>,
        names: Option<&std::collections::BTreeSet<String>>,
        excluded: &std::collections::BTreeSet<String>,
        overrides: &std::collections::BTreeMap<String, ResourceKey>,
        resolve_dependencies: bool,
    ) -> Result<Assembly, AssetError> {
        let definition = root(document);
        let mut components = Vec::new();
        match definition.get("$type").and_then(Value::as_str) {
            Some("entEntityTemplate") => {
                self.expand_components(
                    definition,
                    &mut components,
                    &mut vec![source.clone()],
                    overrides,
                )?;
                let entries = array(definition, "appearances");
                if !entries.is_empty() {
                    let selected = exact(
                        entries,
                        appearance
                            .ok_or_else(|| invalid("entity requires an exact appearance name"))?,
                    )?;
                    let resource = selected
                        .get("appearanceResource")
                        .map(ResourceKey::from_value)
                        .transpose()?
                        .flatten()
                        .ok_or_else(|| invalid("entity appearance has no resource"))?;
                    let resource = redirected(resource, overrides);
                    let app = self.document(&resource)?;
                    let app_name = name(selected, "appearanceName");
                    let selected = exact(array(root(&app), "appearances"), &app_name)?;
                    self.expand_components(
                        selected,
                        &mut components,
                        &mut vec![source.clone(), resource],
                        overrides,
                    )?;
                } else if appearance.is_some_and(|a| !a.is_empty()) {
                    return Err(invalid("simple entity has no selectable appearances"));
                }
            }
            Some("appearanceAppearanceResource") => {
                let selected = exact(
                    array(definition, "appearances"),
                    appearance
                        .ok_or_else(|| invalid("appearance resource requires an exact name"))?,
                )?;
                self.expand_components(
                    selected,
                    &mut components,
                    &mut vec![source.clone()],
                    overrides,
                )?;
            }
            _ => {
                return Err(invalid(
                    "assembly source must be an entity or appearance resource",
                ));
            }
        }
        if let Some(names) = names {
            // A selected visual component still needs the complete native
            // parent/control chain. Keeping only its name silently dropped
            // garment deformation rigs and attachment frames.
            let mut required = names.clone();
            loop {
                let mut added = Vec::new();
                for component in components.iter().filter(|c| required.contains(&c.name)) {
                    for dependency in [&component.parent.component, &component.control.component] {
                        if !required.contains(dependency)
                            && components.iter().any(|c| c.name == *dependency)
                        {
                            added.push(dependency.clone());
                        }
                    }
                }
                if added.is_empty() {
                    break;
                }
                required.extend(added);
            }
            components.retain(|c| required.contains(&c.name));
            for component in components.iter_mut().filter(|c| !names.contains(&c.name)) {
                // Dependency frames are necessary; unrelated donor geometry
                // is not part of the selected garment.
                component.mesh = None;
                let mut template = component.template.as_ref().clone();
                if let Some(data) = template.as_object_mut() {
                    data.remove("mesh");
                    data.remove("morphResource");
                }
                component.template = std::sync::Arc::new(template);
            }
        }
        for component in &mut components {
            component.mesh = component.mesh.take().map(|key| redirected(key, overrides));
            component.rig = component.rig.take().map(|key| redirected(key, overrides));
        }
        let mut rigs = HashMap::new();
        if !resolve_dependencies {
            return Ok(Assembly {
                source: source.clone(),
                appearance: appearance.unwrap_or_default().to_owned(),
                components,
                rigs,
            });
        }
        let mut failures = Vec::new();
        for component in &components {
            if let Some(key) = &component.rig
                && !rigs.contains_key(key)
            {
                match self.rig(key) {
                    Ok(rig) => {
                        rigs.insert(key.clone(), rig);
                    }
                    Err(error) => {
                        failures.push(format!("component {} rig {key}: {error}", component.name))
                    }
                }
            }
        }
        for component in &mut components {
            if component.enabled
                && component.chunk_mask != 0
                && !component.is_collapsed()
                && !excluded.contains(&component.name)
                && component.kind != "entAppearanceProxyMeshComponent"
                && let Some(mesh) = &component.mesh
            {
                match self.mesh_appearance_binding(mesh, &component.mesh_appearance) {
                    Ok(binding) => {
                        component.mesh_appearance = binding.appearance;
                        if binding.non_visual {
                            component.chunk_mask = 0;
                        }
                    }
                    Err(error) => {
                        failures.push(format!("component {} mesh {mesh}: {error}", component.name))
                    }
                }
            }
        }
        if !failures.is_empty() {
            return Err(invalid(failures.join("; ")));
        }
        Ok(Assembly {
            source: source.clone(),
            appearance: appearance.unwrap_or_default().to_owned(),
            components,
            rigs,
        })
    }

    fn expand_components(
        &mut self,
        definition: &Value,
        output: &mut Vec<AssemblyComponent>,
        visiting: &mut Vec<ResourceKey>,
        overrides: &std::collections::BTreeMap<String, ResourceKey>,
    ) -> Result<(), AssetError> {
        if visiting.len() > 64 {
            return Err(invalid("part entity nesting exceeds 64"));
        }
        let start = output.len();
        let handles = handles(definition)?;
        for component in array(definition, "components").iter().chain(
            definition
                .pointer("/compiledData/Data/Chunks")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        ) {
            let component = resolve_handle(component, &handles)?;
            if component_name(component).is_empty() {
                continue;
            }
            let component = decode_component(component, &handles).map_err(|error| {
                invalid(format!("component {}: {error}", component_name(component)))
            })?;
            if let Some(existing) = output.iter_mut().find(|c| c.name == component.name) {
                *existing = component;
            } else {
                output.push(component);
            }
        }
        for part in array(definition, "partsValues") {
            let resource = part
                .get("resource")
                .map(ResourceKey::from_value)
                .transpose()?
                .flatten()
                .ok_or_else(|| invalid("appearance part has no entity resource"))?;
            if visiting.contains(&resource) {
                return Err(invalid(format!("cyclic appearance part {resource}")));
            }
            let resource = redirected(resource, overrides);
            let document = self.document(&resource)?;
            visiting.push(resource);
            let mut part_components = Vec::new();
            self.expand_components(root(&document), &mut part_components, visiting, overrides)?;
            visiting.pop();
            for component in part_components {
                if !output.iter().any(|c| c.name == component.name) {
                    output.push(component);
                }
            }
        }
        for overrides in array(definition, "partsOverrides") {
            for entry in array(unwrap(overrides), "componentsOverrides") {
                let entry = unwrap(entry);
                let target = name(entry, "componentName");
                let Some(component) = output.iter_mut().find(|c| c.name == target) else {
                    // Cross-item overrides can name components outside this assembly.
                    continue;
                };
                if let Some(value) = entry.get("meshAppearance") {
                    text(value)
                        .ok_or_else(|| invalid("invalid mesh appearance override"))?
                        .clone_into(&mut component.mesh_appearance);
                    component.mesh_appearance_requested = Some(component.mesh_appearance.clone());
                }
                if let Some(value) = entry.get("chunkMask") {
                    component.chunk_mask = mask(value)?;
                }
                if let Some(value) = entry.get("isEnabled") {
                    component.enabled = enabled(value)?;
                }
            }
        }
        // Empty authored appearance definitions are metadata-only variants. Do
        // not confuse those with component payloads that failed to decode, or
        // overrides that require an owning entity's component context.
        if output.len() == start
            && start == 0
            && (!array(definition, "components").is_empty()
                || definition
                    .pointer("/compiledData/Data/Chunks")
                    .and_then(Value::as_array)
                    .is_some_and(|chunks| {
                        chunks.iter().any(|chunk| {
                            let chunk = unwrap(chunk);
                            chunk.as_object().is_none_or(|object| {
                                (object.len() != 1 && chunk["$type"] != "entExternalComponent")
                                    || !matches!(
                                        chunk["$type"].as_str(),
                                        Some(
                                            "gameItemObject" | "entEntity" | "entExternalComponent"
                                        )
                                    )
                            })
                        })
                    })
                || !array(definition, "partsOverrides").is_empty())
        {
            return Err(invalid("definition has no named components"));
        }
        Ok(())
    }
}

fn array<'a>(value: &'a Value, field: &str) -> &'a [Value] {
    value
        .get(field)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn exact<'a>(entries: &'a [Value], requested: &str) -> Result<&'a Value, AssetError> {
    let mut matches = entries
        .iter()
        .map(unwrap)
        .filter(|entry| name(entry, "name") == requested);
    let selected = matches.next().ok_or_else(|| {
        invalid(format!(
            "appearance {requested:?} is absent; authored names: {}",
            entries
                .iter()
                .map(|v| name(unwrap(v), "name"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;
    if matches.next().is_some() {
        return Err(invalid(format!("appearance {requested:?} is ambiguous")));
    }
    Ok(selected)
}

pub(crate) fn handles(value: &Value) -> Result<HashMap<String, &Value>, AssetError> {
    let mut result = HashMap::new();
    let mut pending = vec![(value, 0_usize)];
    while let Some((value, depth)) = pending.pop() {
        if depth > 128 {
            return Err(invalid("component handle nesting exceeds 128"));
        }
        match value {
            Value::Object(fields) => {
                if let (Some(id), Some(data)) = (fields.get("HandleId"), fields.get("Data")) {
                    let key = handle_id(id)?;
                    if result
                        .insert(key.clone(), data)
                        .is_some_and(|existing| existing != data)
                    {
                        return Err(invalid(format!("conflicting handle {key}")));
                    }
                }
                pending.extend(fields.values().map(|value| (value, depth + 1)));
            }
            Value::Array(values) => pending.extend(values.iter().map(|value| (value, depth + 1))),
            _ => {}
        }
    }
    Ok(result)
}

fn handle_id(value: &Value) -> Result<String, AssetError> {
    value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_u64().map(|v| v.to_string()))
        .ok_or_else(|| invalid("invalid component handle id"))
}

pub(crate) fn resolve_handle<'a>(
    value: &'a Value,
    handles: &HashMap<String, &'a Value>,
) -> Result<&'a Value, AssetError> {
    if let Some(id) = value.get("HandleRefId") {
        let key = handle_id(id)?;
        if let Some(value) = handles.get(&key) {
            return Ok(value);
        }
        if key == "0" || key == "-1" {
            return Ok(&Value::Null);
        }
        return handles
            .get(&key)
            .copied()
            .ok_or_else(|| invalid(format!("unresolved component handle {key}")));
    }
    Ok(unwrap(value))
}

fn decode_binding(
    value: Option<&Value>,
    handles: &HashMap<String, &Value>,
) -> Result<Binding, AssetError> {
    let Some(value) = value else {
        return Ok(Binding::default());
    };
    let value = resolve_handle(value, handles)?;
    Ok(Binding {
        component: name(value, "bindName"),
        slot: name(value, "slotName"),
    })
}

fn standalone_handles(
    value: &Value,
    handles: &HashMap<String, &Value>,
    emitted: &mut std::collections::HashSet<String>,
    depth: usize,
) -> Result<Value, AssetError> {
    if depth > 128 {
        return Err(invalid("component handle graph exceeds 128"));
    }
    if let Some(id) = value.get("HandleId").or_else(|| value.get("HandleRefId")) {
        let id = handle_id(id)?;
        if !emitted.insert(id.clone()) {
            return Ok(serde_json::json!({"HandleRefId":id}));
        }
        if let Some(data) = value.get("Data").or_else(|| handles.get(&id).copied()) {
            return Ok(
                serde_json::json!({"HandleId":id,"Data":standalone_handles(data,handles,emitted,depth+1)?}),
            );
        }
        if id == "0" || id == "-1" {
            return Ok(Value::Null);
        }
        return Err(invalid(format!("unresolved component handle {id}")));
    }
    match value {
        Value::Object(fields) => Ok(Value::Object(
            fields
                .iter()
                .map(|(key, v)| {
                    Ok((
                        key.clone(),
                        standalone_handles(v, handles, emitted, depth + 1)?,
                    ))
                })
                .collect::<Result<_, AssetError>>()?,
        )),
        Value::Array(values) => Ok(Value::Array(
            values
                .iter()
                .map(|v| standalone_handles(v, handles, emitted, depth + 1))
                .collect::<Result<_, _>>()?,
        )),
        _ => Ok(value.clone()),
    }
}

// CR2W omits constructor defaults. These names are fixed by the native
// vehicleChassisComponent and entPlaceholderComponent constructors, not aliases
// inferred from references or a particular vehicle's layout.
fn redirected(
    key: ResourceKey,
    overrides: &std::collections::BTreeMap<String, ResourceKey>,
) -> ResourceKey {
    overrides
        .get(&key.native_hash().to_string())
        .cloned()
        .unwrap_or(key)
}

fn component_name(value: &Value) -> &str {
    if let Some(name) = value.get("name") {
        return text(name).unwrap_or_default();
    }
    match value.get("$type").and_then(Value::as_str) {
        Some("vehicleChassisComponent") => "Chassis",
        Some("entPlaceholderComponent") => "entPlaceholderComponent",
        _ => "",
    }
}

fn decode_component(
    value: &Value,
    handles: &HashMap<String, &Value>,
) -> Result<AssemblyComponent, AssetError> {
    let template = standalone_handles(value, handles, &mut std::collections::HashSet::new(), 0)?;
    let mut local =
        Transform::decode(value.get("localTransform").unwrap_or(&Value::Null))?.matrix()?;
    let scale = vector(value.get("visualScale"), [1.0; 3])?;
    if !scale.contains(&0.0) {
        local *= Mat4::from_scale(scale.into());
    }
    let slots = array(value, "slots").iter().map(|slot| {
        let slot = unwrap(slot);
        let transform = serde_json::json!({"Translation": slot.get("relativePosition"), "Rotation": slot.get("relativeRotation")});
        Ok(Slot { name: name(slot, "slotName"), bone: name(slot, "boneName"), transform: Transform::decode(&transform)? })
    }).collect::<Result<Vec<_>, AssetError>>()?;
    Ok(AssemblyComponent {
        template: std::sync::Arc::new(template),
        name: component_name(value).to_owned(),
        kind: name(value, "$type"),
        local,
        parent: decode_binding(value.get("parentTransform"), handles)?,
        control: decode_binding(
            value
                .get("controlBinding")
                .or_else(|| value.get("skinning")),
            handles,
        )?,
        rig: value
            .get("rig")
            .map(ResourceKey::from_value)
            .transpose()?
            .flatten(),
        mesh: value
            .get("morphResource")
            .or_else(|| value.get("mesh"))
            .map(ResourceKey::from_value)
            .transpose()?
            .flatten(),
        mesh_appearance: value
            .get("meshAppearance")
            .and_then(text)
            .unwrap_or("default")
            .to_owned(),
        mesh_appearance_requested: value
            .get("meshAppearance")
            .and_then(text)
            .map(str::to_owned),
        chunk_mask: value
            .get("chunkMask")
            .map(mask)
            .transpose()?
            .unwrap_or(u64::MAX),
        enabled: value
            .get("isEnabled")
            .map(enabled)
            .transpose()?
            .unwrap_or(true),
        slots,
    })
}

fn mask(value: &Value) -> Result<u64, AssetError> {
    value
        .as_u64()
        .or_else(|| text(value).and_then(|s| s.parse().ok()))
        .ok_or_else(|| invalid("invalid chunk mask"))
}

fn enabled(value: &Value) -> Result<bool, AssetError> {
    value
        .as_bool()
        .or_else(|| value.as_u64().map(|v| v != 0))
        .ok_or_else(|| invalid("invalid enable flag"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collapsed_visual_scale_keeps_the_attachment_frame() {
        let value = serde_json::json!({"$type":"entMeshComponent","name":"cargo",
            "visualScale":{"X":0.0,"Y":1.0,"Z":1.0},
            "localTransform":{"Position":{"X":2.0}}});
        let component = decode_component(&value, &handles(&value).unwrap()).unwrap();
        assert!(component.is_collapsed());
        assert_eq!(component.local.w_axis.x, 2.0);
        assert!(component.local.inverse().is_finite());
        assert_eq!(component.template["visualScale"]["X"], 0.0);
    }

    #[test]
    fn omitted_native_component_names_keep_their_authored_transforms() {
        let chassis = json!({"$type":"vehicleChassisComponent",
            "localTransform":{"Position":{"Z":0.5}},
            "parentTransform":{"Data":{"bindName":"vehicle_root"}}});
        let component = decode_component(&chassis, &handles(&chassis).unwrap()).unwrap();
        assert_eq!(component.name, "Chassis");
        assert_eq!(component.local.w_axis.z, 0.5);
        assert_eq!(component.parent.component, "vehicle_root");
        assert_eq!(
            component_name(&json!({"$type":"entPlaceholderComponent"})),
            "entPlaceholderComponent"
        );
        assert_eq!(component_name(&json!({"$type":"unknownComponent"})), "");
        assert_eq!(
            component_name(&json!({"$type":"vehicleChassisComponent","name":"custom"})),
            "custom"
        );
    }

    #[test]
    fn component_handles_hashes_and_exact_appearances_are_preserved() {
        let value = json!({"name":"mesh", "$type":"entSkinnedMeshComponent", "mesh":{"DepotPath":{"$value":"18446744073709551614"}}, "chunkMask":"18446744073709551615", "parentTransform":{"HandleRefId":"7"}, "binding":{"HandleId":"7", "Data":{"bindName":"rig","slotName":"Hand"}}});
        let handles = handles(&value).unwrap();
        let component = decode_component(&value, &handles).unwrap();
        assert_eq!(component.mesh, Some(ResourceKey::Hash(u64::MAX - 1)));
        assert_eq!(component.chunk_mask, u64::MAX);
        assert_eq!(component.parent.component, "rig");
        assert_eq!(component.parent.slot, "Hand");
        assert!(exact(&[json!({"name":"red"})], "default").is_err());
        assert!(exact(&[json!({"name":"red"}), json!({"name":"red"})], "red").is_err());
    }
}
