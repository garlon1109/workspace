use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=reef.h");

    let bindings = bindgen::Builder::default()
        .header("reef.h")
        .clang_args(["-x", "c++"])
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("reef.*")
        .allowlist_function("cuCtxSetCurrent")
        .allowlist_type("CUcontext")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
