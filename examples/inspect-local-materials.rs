//! Inspect embedded dependencies inside native mesh-local material documents.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: inspect-local-materials MESH_JSON GAME OUTPUT".into());
    }
    let document: Value = serde_json::from_slice(&fs::read(&args[0])?)?;
    let schema = ghostline_red::schema::RedSchema::load(Some(Path::new(&args[1])), None)?;
    let root = &document["Data"]["RootChunk"];
    let buffer = &root["localMaterialBuffer"];
    let bytes = STANDARD.decode(
        buffer["rawData"]["Bytes"]
            .as_str()
            .ok_or("mesh has no local material bytes")?,
    )?;
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    for (index, header) in buffer["rawDataHeaders"]
        .as_array()
        .ok_or("no material headers")?
        .iter()
        .enumerate()
    {
        let offset = usize::try_from(header["offset"].as_u64().unwrap_or(0))?;
        let size = usize::try_from(header["size"].as_u64().ok_or("no material size")?)?;
        let material = ghostline_red::codec::decode_wkit_from_bytes(
            bytes
                .get(offset..offset.checked_add(size).ok_or("material overflow")?)
                .ok_or("material bounds")?,
            &schema,
        )?;
        fs::write(
            out.join(format!("{index}.json")),
            serde_json::to_vec_pretty(&material)?,
        )?;
        println!(
            "{}",
            json!({"index":index,"embedded":material["Data"]["EmbeddedFiles"],"base":material["Data"]["RootChunk"]["baseMaterial"]})
        );
    }
    Ok(())
}
