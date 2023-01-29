use std::env;

use cbindgen::{Config, ExportConfig, ItemType};

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(Config {
            language: cbindgen::Language::C,
            cpp_compat: true,
            pragma_once: true,
            export: ExportConfig {
                item_types: vec![ItemType::Functions, ItemType::Typedefs],
                ..Default::default()
            },
            ..Default::default()
        })
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("../target/client.h");
}
