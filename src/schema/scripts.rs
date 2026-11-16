//! Script-defined reflection from the installed, unmodded script bundle.
use super::{RedClass, RedProperty, RedSchema, SchemaError};
use redscript_io::{Definition, ScriptBundle, TypeIndex, TypeKind};
use std::collections::{BTreeMap, BTreeSet};

fn name(bundle: &ScriptBundle<'_>, index: redscript_io::CNameIndex) -> Result<String, SchemaError> {
    bundle
        .get_item(index)
        .map(std::string::ToString::to_string)
        .ok_or_else(|| SchemaError::Invalid("invalid script name reference".into()))
}

fn type_name(
    bundle: &ScriptBundle<'_>,
    index: TypeIndex,
    visited: &mut BTreeSet<String>,
) -> Result<String, SchemaError> {
    let ty = bundle
        .get_item(index)
        .ok_or_else(|| SchemaError::Invalid("invalid script type reference".into()))?;
    let own_name = name(bundle, ty.name())?;
    if !visited.insert(own_name.clone()) {
        return Err(SchemaError::Invalid(format!(
            "cyclic script type: {own_name}"
        )));
    }
    let result = match ty.kind() {
        TypeKind::Primitive | TypeKind::Class => own_name.clone(),
        TypeKind::Ref(inner) => format!("handle:{}", type_name(bundle, inner, visited)?),
        TypeKind::WeakRef(inner) => format!("whandle:{}", type_name(bundle, inner, visited)?),
        TypeKind::Array(inner) => format!("array:{}", type_name(bundle, inner, visited)?),
        TypeKind::StaticArray { element_type, size } => {
            format!("[{size}]{}", type_name(bundle, element_type, visited)?)
        }
        TypeKind::ScriptRef(inner) => format!("script_ref:{}", type_name(bundle, inner, visited)?),
    };
    visited.remove(&own_name);
    Ok(result)
}

pub(super) fn read(bytes: &[u8]) -> Result<RedSchema, SchemaError> {
    let bundle = ScriptBundle::from_bytes(bytes)
        .map_err(|e| SchemaError::Invalid(format!("invalid game script bundle: {e:?}")))?;
    let mut schema = RedSchema::default();
    for definition in bundle.definitions() {
        match definition {
            Definition::Class(class) => {
                let mut properties = BTreeMap::new();
                for (ordinal, index) in class.fields().iter().enumerate() {
                    let field = bundle.get_item(*index).ok_or_else(|| {
                        SchemaError::Invalid("invalid script field reference".into())
                    })?;
                    let red_type = type_name(&bundle, field.type_(), &mut BTreeSet::new())?;
                    properties.insert(
                        name(&bundle, field.name())?,
                        RedProperty {
                            cs_type: red_type.clone(),
                            red_type: Some(red_type),
                            ordinal: Some(u32::try_from(ordinal).map_err(|_| {
                                SchemaError::Invalid("script field ordinal overflow".into())
                            })?),
                            offset: None,
                            flags: None,
                        },
                    );
                }
                let base = class
                    .base()
                    .map(|index| {
                        bundle
                            .get_item(index)
                            .ok_or_else(|| {
                                SchemaError::Invalid("invalid script parent reference".into())
                            })
                            .and_then(|parent| name(&bundle, parent.name()))
                    })
                    .transpose()?;
                schema.classes.insert(
                    name(&bundle, class.name())?,
                    RedClass {
                        base,
                        properties,
                        flags: None,
                        size: None,
                        alignment: None,
                    },
                );
            }
            Definition::Enum(value) => {
                schema.enums.insert(name(&bundle, value.name())?);
            }
            _ => {}
        }
    }
    Ok(schema)
}
