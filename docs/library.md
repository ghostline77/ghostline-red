# Library reference

## Dependency and API documentation

For a project beside this checkout, add this dependency to its `Cargo.toml`:

```toml
[dependencies]
ghostline-red = { git = "https://github.com/ghostline77/ghostline-red" }
```

Generate the Rust API reference from the ghostline-red repository root:

```powershell
cargo doc --no-deps --open
```

## Resource decoding

This example decodes a loose CR2W file using bundled reflection:

```rust
use ghostline_red::{codec, schema::RedSchema};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = RedSchema::bundled()?;
    let bytes = std::fs::read("input.ent")?;
    let document = codec::decode_wkit_from_bytes(&bytes, &schema)?;
    println!("{document}");
    Ok(())
}
```

| API | Inputs | Result |
| --- | --- | --- |
| `schema::RedSchema::bundled` | None | Embedded serialization definitions and game reflection. |
| `schema::RedSchema::load` | `game_root: Option<&Path>`, `override_path: Option<&Path>`; either may be `None`. | Bundled reflection with the requested overlays. |
| `cr2w::inspect_bytes` | Required `bytes: &[u8]`. | Header, export and buffer metadata. |
| `codec::decode_wkit_from_bytes` | Required `bytes: &[u8]`, `schema: &RedSchema`. | Recursive JSON resource document. |
| `writer::write` | Required `input: &Path`, `output: &Path`, `schema: &RedSchema`. | A newly generated CR2W resource. |
| `writer::write_with_red_schema` | Required `json_path: &Path`, `template_path: &Path`, `output_path: &Path`, `schema: &RedSchema`. | A CR2W resource rebuilt with the explicit template. |

See [reflection metadata](../data/README.md) for overlay precedence.

## Archive access and caching

`material::ArchiveSet::open` recursively indexes an archive directory. Pass
`game_root.join("archive/pc")` to select the installed PC archives.
`ArchiveSet::from_paths` indexes an explicit list instead.

Archive paths are sorted; later paths win duplicate resource hashes. This is
the library's deterministic source-selection rule, not the game's runtime
mod-loading policy. `r6/c77` is outside `archive/pc`; include those archives
explicitly with `from_paths` when they belong in your source selection.

`ArchiveSet::read_resource` reads a depot path. `read_resource_hash` reads an exact
64-bit resource identity. `content_identity` fingerprints the selected resource
identities and stored payload hashes. Reopen the archive set after its source
archives change.

`assets::AssetReader::new(&archives, &schema)` reuses that archive index and caches
decoded documents. `document(&ResourceKey)` returns a shared `Arc<Value>`.
`ResourceKey` accepts either a depot path or a native hash. Use `with_cache` and
`assets::DocumentCache` to share decoded documents across readers.

## Resource modules

| Module | Resources and operations |
| --- | --- |
| [`archive`](../src/archive.rs) | Archive indexes, extraction and packing. |
| [`assets`](../src/assets/mod.rs) | Entity and appearance assembly, rigs, tables, customization and native resource identities. |
| [`mesh`](../src/mesh.rs) | Vertex/index streams, render chunks, skin weights and inverse bind matrices. |
| [`animation`](../src/animation.rs) | Named clips, compressed-key and SIMD animation buffers, tracks and motion extraction. |
| [`morph`](../src/morph.rs) | Base geometry and sparse morph targets. |
| [`material`](../src/material.rs) | Archive-backed material bindings and dependency export. |
| [`pbr`](../src/pbr.rs) | Preview material parameters, baked textures and chunk-to-material mappings. |
| [`tweakdb`](../src/tweakdb.rs) | Compiled records, flats, queries, record types and optional name hints. |
| [`traffic`](../src/traffic.rs) | Traffic-lane geometry and connectivity. |
| [`world`](../src/world.rs) | Native world collision primitives. |
| [`texture`](../src/texture.rs) | UI textures and single-image inventory atlases. |
| [`facial`](../src/facial.rs) | Facial animation-set generation. |
| [`kraken`](../src/kraken.rs) | Raw Kraken compression and decompression. |

## Geometry and animation conventions

- Geometry and inverse bind matrices use Y-up glTF coordinates. Rig transforms
  and animation channels retain native RED coordinates for the caller to convert.
- Mesh decoding preserves original render-chunk indices and normalized four- or
  eight-influence skin weights.
- Animation sampling interpolates translation and scale linearly, and rotation
  with quaternion slerp. Additive animation requires an explicit base pose.

## Material conversion

Material conversion prepares static previews. Converted material notes record
unsupported features, substitutions and approximations.

| Feature | Preview behavior |
| --- | --- |
| Skin, eyes and hair | Supported tint, texture, gradient and coverage inputs are converted. Subsurface scattering, refraction, hair anisotropy and dynamic detail can be approximated or omitted. |
| Decals | Masks and lookup colors are baked. Projection depth and blending into the receiving surface's normals are not simulated. |
| Glass | Label coverage and the tinted transparent region remain separate. Transmission uses an alpha approximation. |
| Dynamic surfaces | Supported authored animation frames can be sampled for a static mesh. Procedural runtime behavior is not reproduced. |

Native byte `Color` tints are converted from sRGB. `Vector4` tints retain their
linear values.
