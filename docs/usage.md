# CLI workflows

Run these examples from the repository root in PowerShell. Replace the example
input paths, depot paths and appearance names with those of your resources.

```powershell
$red = './target/release/ghostline-red.exe'
$game = 'C:/Games/Cyberpunk 2077'
```

On Linux, use `./target/release/ghostline-red` for `$red`.
See the [build instructions](../README.md#build) if the executable is missing.

## Archives

### List and extract files

List an archive's entries as JSON:

```powershell
& $red archive-list 'example.archive' --json
```

Extract every file, or select one exact depot path:

```powershell
& $red extract 'example.archive' --output 'extracted'
& $red extract 'example.archive' --output 'extracted' --path 'mod/example/character.ent'
```

If entries have hashes instead of names, supply a loose depot tree whose relative
paths can resolve those hashes:

```powershell
& $red archive-list 'example.archive' --paths-root 'source/archive' --json
& $red extract 'example.archive' --output 'extracted' --paths-root 'source/archive'
```

### Pack a depot tree

Place resources beneath the directory that represents the depot root. For
example, `source/archive/mod/example/character.ent` has the depot path
`mod\example\character.ent`.

```powershell
& $red pack 'source/archive' --output 'example.archive'
& $red archive-list 'example.archive' --json
```

The output is an archive file. Its parent directory must already exist.
Packing does not install the archive or create mod registration files.

## CR2W resources

Inspect a binary resource's header and tables, then convert it to recursive
WolvenKit-shaped JSON:

```powershell
& $red cr2w-inspect 'input.ent' --json
& $red cr2w-serialize 'input.ent' 'input.ent.json'
```

Write supported resources directly from authored JSON:

```powershell
& $red cr2w-deserialize 'input.ent.json' 'output.ent'
```

If you need to preserve data from an existing binary while applying edits,
pass that binary explicitly as the template:

```powershell
& $red cr2w-deserialize 'input.ent.json' 'output.ent' --template 'input.ent'
```

To use metadata from a particular installation, add `--game`:

```powershell
& $red cr2w-serialize 'input.ent' 'input.ent.json' --game $game
```

No schema path is needed. See [reflection metadata](../data/README.md) for
advanced overrides and metadata precedence.

## Batch processing

Batch commands index installed archives once per invocation. Relative manifest
and output paths are resolved from the working directory, not the manifest's
directory. Use distinct output paths for each job.

### Convert archive resources to JSON

Save a manifest as `resources.json`:

```json
{
  "jobs": [
    {
      "resource": "mod/example/character.ent",
      "output": "decoded/character.ent.json"
    }
  ]
}
```

Run the batch:

```powershell
& $red cr2w-serialize-batch 'resources.json' --game $game --report 'resource-report.json'
```

The report contains each job's `resource`, `output` and nullable `error`.
The command exits with an error if any job fails, after writing the report.

### Export mesh appearances

Save a manifest as `meshes.json`:

```json
{
  "jobs": [
    {
      "mesh": "mod/example/jacket.mesh",
      "appearance": "default",
      "output": "exports/jacket.glb"
    }
  ]
}
```

For geometry alone, omit `appearance` if it is not needed and use `--geometry-only`:

```powershell
& $red mesh-export-batch 'meshes.json' --game $game --geometry-only --report 'mesh-report.json'
```

For material dependencies and a material sidecar, supply `--material-repo`:

```powershell
& $red mesh-export-batch 'meshes.json' --game $game --material-repo 'materials' --report 'mesh-report.json'
```

To bake supported materials into glTF PBR textures, add `--pbr`. Every PBR job
must specify an existing appearance name:

```powershell
& $red mesh-export-batch 'meshes.json' --game $game --material-repo 'materials' --pbr --pbr-size 1024 --report 'mesh-report.json'
```

Each report entry contains `mesh`, `appearance`, `output`, `error` and
`material_error`. A material export failure without `--pbr` is reported as a
material warning; a geometry failure or PBR failure makes the job fail.

Both batch commands accept `--threads`. The default, `0`, uses the available
logical CPUs. For example, append `--threads 16` to set 16 worker threads.

## Mesh export

Export an extracted mesh to binary glTF:

```powershell
& $red mesh-export 'input.mesh' --output 'output.glb'
```

The single-mesh command exports the first render LOD by default. Add `--all-lods`
to include every LOD.

Export a named appearance with supported PBR materials:

```powershell
& $red mesh-export 'input.mesh' --output 'output.glb' --game $game --appearance 'default' --material-repo 'materials' --pbr --pbr-size 1024
```

Material export writes an adjacent `output.Material.json` sidecar. If you already
have a GLB and its sidecar, bake the selected appearance with:

```powershell
& $red pbr-bake 'output.glb' --sidecar 'output.Material.json' --appearance 'default' --size 1024
```

See [material conversion limits](library.md#material-conversion) for supported
approximations and omissions.

## TweakDB inspection

Inspect a record in an extracted version 8/parser 4 TweakDB cache. Repeat
`--property` to select multiple fields:

```powershell
& $red tweakdb-inspect 'tweakdb_ep1.bin' --record 'Items.Vest_17_basic_02' --property 'appearanceName' --property 'itemType'
```

Use `--record-class` instead of `--record` to select records by reflected class,
including derived classes. The record argument also accepts a decimal TweakDBID.

## Compression

Encode a file as a raw Kraken stream:

```powershell
& $red kraken-encode 'input.bin' 'encoded.kraken'
```

Decode it using the exact uncompressed size in bytes. This example assumes the
original input was 1,048,576 bytes:

```powershell
& $red kraken-decode 'encoded.kraken' 'decoded.bin' --size 1048576
```
