//! Generate and verify native facial animation sets from curve files.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use ghostline_red::{
    assets::{AssetReader, ResourceKey},
    material::ArchiveSet,
    schema::RedSchema,
};
use serde_json::Value;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    rig: String,
    #[arg(long)]
    curves: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let options = Options::parse();
    let schema = RedSchema::load(Some(&options.game), options.schema.as_deref())?;
    let archives = ArchiveSet::open(&options.game)?;
    let mut reader = AssetReader::new(&archives, &schema);
    let rig = reader.document(&ResourceKey::Path(options.rig.clone()))?;
    let report: Value = serde_json::from_slice(&fs::read(options.curves.join("report.json"))?)?;
    ensure!(
        report["failures"].as_array().is_some_and(Vec::is_empty),
        "facial generation has failures"
    );
    let clips = report["jobs"]
        .as_array()
        .context("curve report jobs")?
        .iter()
        .map(|job| {
            Ok(serde_json::from_slice(&fs::read(
                job["curves"].as_str().context("curve path")?,
            )?)?)
        })
        .collect::<Result<Vec<Value>>>()?;
    let document = ghostline_red::facial::animation_set(&options.rig, &rig, &clips)?;
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = tempfile::NamedTempFile::new()?;
    fs::write(raw.path(), serde_json::to_vec(&document)?)?;
    ghostline_red::writer::write(raw.path(), &options.output, &schema)?;
    let decoded = ghostline_red::codec::decode_wkit_with_red_schema(&options.output, &schema)?;
    ensure!(
        decoded["Data"]["RootChunk"]["animations"]
            .as_array()
            .context("native animations")?
            .len()
            == clips.len(),
        "facial animation readback count"
    );
    println!("Verified {} generated native facial clips", clips.len());
    Ok(())
}
