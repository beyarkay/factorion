//! Bakes a hash of the crate's sources into the build as `SOURCE_HASH`, so
//! `factorion.py` can tell a stale installed extension from a current one.
//! `factorion._factorion_rs_source_hash` must compute the identical hash.

use sha2::{Digest, Sha256};
use std::path::Path;

fn rust_files(dir: &Path, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path.to_str().unwrap().replace('\\', "/"));
        }
    }
}

fn main() {
    let mut paths = vec!["Cargo.toml".into(), "Cargo.lock".into(), "build.rs".into()];
    rust_files(Path::new("src"), &mut paths);
    paths.sort();
    let mut hasher = Sha256::new();
    for path in &paths {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        hasher.update(std::fs::read(path).unwrap());
        hasher.update([0]);
        println!("cargo:rerun-if-changed={path}");
    }
    // A directory entry makes cargo notice newly added files too.
    println!("cargo:rerun-if-changed=src");
    println!(
        "cargo:rustc-env=FACTORION_RS_SOURCE_HASH={:x}",
        hasher.finalize()
    );
}
