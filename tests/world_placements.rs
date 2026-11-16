//! Cooked placements select their own native mesh and transform range.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use ghostline_red::world::placement_node;
use serde_json::json;

#[test]
fn instanced_placements_keep_distinct_meshes_and_transform_ranges() {
    let mut tail = vec![0, 2];
    for mesh in [42_u64, 43] {
        let mut entry = vec![0; 64];
        entry[32..40].copy_from_slice(&mesh.to_le_bytes());
        entry[40..48].copy_from_slice(&99_u64.to_le_bytes());
        tail.extend(entry);
    }
    tail.push(2);
    for value in [1_u16, 2, 3, 1] {
        tail.extend(value.to_le_bytes());
    }
    let mut node = json!({"$type":"worldInstancedMeshNode","version":2,
        "$rawTail":STANDARD.encode(&tail),"cookedWorldTransformsBuffer":{"Bytes":STANDARD.encode(vec![0;192])}});
    let first = placement_node(&node, &json!({"InstanceIndex":0})).unwrap();
    let second = placement_node(&node, &json!({"InstanceIndex":1})).unwrap();
    assert_eq!(first["mesh"]["DepotPath"]["$value"], "42");
    assert_eq!(second["mesh"]["DepotPath"]["$value"], "43");
    assert_eq!(first["worldTransformsBuffer"]["startIndex"], 1);
    assert_eq!(first["worldTransformsBuffer"]["numElements"], 2);
    assert_eq!(second["worldTransformsBuffer"]["startIndex"], 3);
    assert_eq!(second["worldTransformsBuffer"]["numElements"], 1);
    assert!(placement_node(&node, &json!({"InstanceIndex":2})).is_err());
    node["cookedWorldTransformsBuffer"]["Bytes"] = json!(STANDARD.encode(vec![0; 48]));
    assert!(placement_node(&node, &json!({"InstanceIndex":1})).is_err());
    tail.pop();
    node["$rawTail"] = json!(STANDARD.encode(tail));
    assert!(placement_node(&node, &json!({"InstanceIndex":0})).is_err());
}

#[test]
fn static_placements_keep_native_class_and_appearance_hash() {
    let mut tail = vec![33];
    tail.extend(42_u64.to_le_bytes());
    tail.extend(99_u64.to_le_bytes());
    tail.extend([0; 17]);
    let node = json!({"$type":"worldCookedStaticMeshNode","$rawTail":STANDARD.encode(tail)});
    let resolved = placement_node(&node, &json!({"InstanceIndex":0})).unwrap();
    assert_eq!(resolved["$type"], "worldCookedStaticMeshNode");
    assert_eq!(resolved["meshAppearance"]["$value"], "99");
    assert_eq!(resolved["$rawTail"], node["$rawTail"]);
    assert!(placement_node(&node, &json!({"InstanceIndex":1})).is_err());
}

#[test]
fn cooked_collision_placements_select_their_own_actor_table() {
    use ghostline_red::world::{CollisionBox, collision_shapes, encode_boxes};
    let mut tail = vec![0x60, 0x01]; // 96 descriptor bytes in native signed VLQ.
    let mut payload = Vec::new();
    for x in [1., 5.] {
        let ordinary = encode_boxes(
            &[CollisionBox {
                position: [x, 2., 3.],
                rotation: [0., 0., 0., 1.],
                size: [2., 2., 2.],
            }],
            "test",
        )
        .unwrap();
        let mut descriptor = vec![0; 48];
        descriptor[24..28].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        for (i, name) in [
            "numActors",
            "numShapeInfos",
            "numShapePositions",
            "numShapeRotations",
            "numScales",
            "numMaterials",
            "numPresets",
            "numShapeIndices",
            "numMaterialIndices",
        ]
        .iter()
        .enumerate()
        {
            descriptor[28 + i * 2..30 + i * 2]
                .copy_from_slice(&(ordinary[*name].as_u64().unwrap() as u16).to_le_bytes());
        }
        tail.extend(descriptor);
        payload.extend(
            STANDARD
                .decode(ordinary["compiledData"]["Bytes"].as_str().unwrap())
                .unwrap(),
        );
    }
    let cooked = json!({"$type":"worldCookedCollisionNode","$rawTail":STANDARD.encode(tail),
        "compiledData":{"Bytes":STANDARD.encode(payload)}});
    let second = placement_node(&cooked, &json!({"InstanceIndex":1})).unwrap();
    let shapes = collision_shapes(&second).unwrap();
    assert_eq!(shapes.len(), 1);
    assert!((shapes[0].position[0] - 5.).abs() < f32::EPSILON);
    let legacy =
        json!({"$type":"worldInstancedMeshNode","worldTransformsBuffer":{"numElements":3}});
    assert_eq!(placement_node(&legacy, &json!({})).unwrap(), legacy);
}
