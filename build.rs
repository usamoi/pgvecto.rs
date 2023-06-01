fn main() {
    cxx_build::bridge("src/bridge/mod.rs")
        .file("src/bridge/bindings.cpp")
        .flag_if_supported("-std=c++20")
        .compile("cxxbridge-hnswlib");

    println!("cargo:rerun-if-changed=src/bridge/mod.rs");
    println!("cargo:rerun-if-changed=src/bridge/bindings.cpp");
    println!("cargo:rerun-if-changed=src/bridge/hnswlib.h");
}