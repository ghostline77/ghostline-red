# ghostline-red

Rust library and command-line tools for Cyberpunk 2077 archives, CR2W resources,
meshes, materials, animation and localization. Archive compression and
decompression use the native Rust Kraken codec.

| Task | Documentation |
| --- | --- |
| Extract or pack an archive | [Archive workflows](docs/usage.md#archives) |
| Convert a resource to or from JSON | [CR2W workflows](docs/usage.md#cr2w-resources) |
| Process archive resources or meshes in batches | [Batch workflows](docs/usage.md#batch-processing) |
| Export a mesh with materials | [Mesh export](docs/usage.md#mesh-export) |
| Use the Rust library | [Library reference](docs/library.md) |
| Inspect or update bundled reflection | [Reflection metadata](data/README.md) |

## Build

Build from the repository root with a stable Rust toolchain:

```powershell
cargo build --release --locked
```

On Windows, the main executable is `target/release/ghostline-red.exe`. On Linux,
it is `target/release/ghostline-red`.

## CLI setup

The examples use PowerShell from the repository root. Set the executable path
once, then replace example input paths with your own files:

```powershell
$red = './target/release/ghostline-red.exe'
& $red --help
& $red archive-list 'example.archive' --json
```

Resource conversion uses bundled reflection metadata. No schema generation,
schema path or WolvenKit installation is required.

Commands that read installed archives use `--game` with the game installation
directory. They derive `archive/pc` from that path:

```powershell
$game = 'C:/Games/Cyberpunk 2077'
& $red mesh-export-batch 'meshes.json' --game $game --geometry-only --report 'mesh-report.json'
```

For loose-file resource conversion, `--game` is optional. It supplements the
bundled metadata with metadata from that installation. `--schema` is an optional
advanced override on commands that expose it; see [reflection metadata](data/README.md).

Use `& $red <command> --help` to look up a command's arguments.

## Development checks

```powershell
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

Fixture-dependent tests are marked ignored and list their required inputs.
Binary readback and archive verification check serialized data; they do not
establish in-game behavior or console compatibility.
