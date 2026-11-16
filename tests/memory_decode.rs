//! File and memory decoding share native table and handle semantics.

use std::fs;

use ghostline_red::{codec, cr2w, schema::RedSchema, writer};
use serde_json::json;

#[test]
fn cooked_world_node_appendices_survive_native_roundtrip() {
    for class in [
        "worldCookedAcousticSectorNode",
        "worldCookedGenericProxyMeshNode",
        "worldCookedStaticDecalNode",
        "worldCookedStaticMeshNode",
        "worldCookedBendedMeshNode",
        "worldCookedCableMeshNode",
        "worldCookedFoliageDestructionNode",
    ] {
        let schema = RedSchema::from_slice(
            &serde_json::to_vec(&json!({class:{"base":null,"properties":{}}})).unwrap(),
        )
        .unwrap();
        let folder = tempfile::tempdir().unwrap();
        let input = folder.path().join("cooked.json");
        let output = folder.path().join("cooked.cr2w");
        let document = json!({"Data":{"RootChunk":{
            "$type":class,"$rawTail":"AQIDBAUGBwg="
        }}});
        fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
        writer::write(&input, &output, &schema).unwrap();
        let decoded = codec::decode_wkit_from_bytes(&fs::read(&output).unwrap(), &schema).unwrap();
        assert_eq!(
            decoded["Data"]["RootChunk"]["$rawTail"],
            document["Data"]["RootChunk"]["$rawTail"]
        );
    }
}

#[test]
fn memory_and_file_decoding_preserve_identical_tables_and_handles() {
    let schema = RedSchema::from_slice(
        br#"{
        "Example":{"base":null,"properties":{
            "enabled":{"cs_type":"CBool","ordinal":0},
            "children":{"cs_type":"CArray<CHandle<Example>>","ordinal":1}
        }}
    }"#,
    )
    .unwrap();
    let folder = tempfile::tempdir().unwrap();
    let input = folder.path().join("input.json");
    let output = folder.path().join("fixture.cr2w");
    let document = json!({"Data":{"RootChunk":{"$type":"Example","enabled":true,
        "children":[{"HandleId":"42","Data":{"$type":"Example","enabled":false,"children":[]}}]}}});
    fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    writer::write(&input, &output, &schema).unwrap();
    let bytes = fs::read(&output).unwrap();
    let disk_tables = serde_json::to_value(cr2w::inspect(&output).unwrap()).unwrap();
    let memory_tables = serde_json::to_value(cr2w::inspect_bytes(&bytes).unwrap()).unwrap();
    assert_eq!(disk_tables, memory_tables);
    let mut disk = codec::decode_wkit_with_red_schema(&output, &schema).unwrap();
    let memory = codec::decode_wkit_from_bytes(&bytes, &schema).unwrap();
    disk["Header"]["ArchiveFileName"] = json!("");
    assert_eq!(disk, memory);
    assert_eq!(
        memory["Data"]["RootChunk"]["children"][0]["Data"]["enabled"],
        json!(false)
    );
    for size in [0, 4, 16, bytes.len() - 1] {
        assert!(codec::decode_wkit_from_bytes(&bytes[..size], &schema).is_err());
    }
}
