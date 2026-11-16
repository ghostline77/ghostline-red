//! Expose the resolved decoder source to downstream rendering cache fingerprints.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::metadata=source_dir={}", env!("CARGO_MANIFEST_DIR"));
}
