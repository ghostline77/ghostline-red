# Reflection metadata

The library and CLI embed their reflection metadata. Reading and writing
resources requires neither schema generation nor a schema path. WolvenKit is
not a runtime dependency.

## Bundled files

| File | Contents | Source |
| --- | --- | --- |
| [`schema.json`](schema.json) | Game classes, properties, enums, aliases and source fingerprints. | The original game script bundle and REDmod metadata. |
| [`serialization.json`](serialization.json) | Serialized property names, type notation, ordering and custom format fields. | WolvenKit-derived serialization definitions. |

The bundled schema combines the game snapshot with the serialization definitions,
then reapplies the snapshot's native metadata. This retains serialized fields
that static game reflection does not describe.

## Loading and overrides

| Entry point | Behavior |
| --- | --- |
| `RedSchema::bundled()` | Loads the embedded files without filesystem access. |
| `RedSchema::load(Some(game_root), None)` | Adds installed script definitions without replacing known serialization definitions, then overlays installed native metadata. |
| `RedSchema::load(Some(game_root), Some(path))` | Applies the supplied schema after the bundled and installed metadata. |
| `RedSchema::bundled_provenance()` | Returns fingerprints for the embedded game inputs and serialization definitions. |

Installed metadata is read from `r6/cache/final.redscripts` and
`tools/redmod/metadata.json`. Missing optional files leave the bundled metadata
in effect; malformed files return an error. The loader does not read modded script
caches.

For a CLI override on a command that supports `--schema`:

```powershell
./target/release/ghostline-red.exe cr2w-serialize 'input.ent' 'output.json' --schema 'custom-schema.json'
```

Commands whose help describes schema sources in increasing precedence order
accept repeated `--schema` options. Later sources override earlier ones.

## Update the game snapshot

This procedure is for maintaining the embedded game snapshot. It requires an
installation containing the original script bundle and REDmod metadata.

From the repository root, generate the snapshot:

```powershell
cargo run --release --locked --bin ghostline-red -- schema-generate data/schema.json --game 'C:/Games/Cyberpunk 2077'
```

The generator updates `schema.json` only. It records SHA-1 fingerprints of the
script bundle, REDmod metadata and the game executable when present. It does not
update `serialization.json` or launch the game.

Rebuild to embed the updated data, then check schema coverage:

```powershell
cargo build --release --locked
cargo test --locked --lib schema::tests
```
