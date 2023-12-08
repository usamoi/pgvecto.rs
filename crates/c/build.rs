fn main() {
    println!("rerun-if-changed:src/c.h");
    println!("rerun-if-changed:src/c.c");
    cc::Build::new()
        .compiler("/usr/bin/clang-16")
        .file("./src/c.c")
        .opt_level(3)
        .compile("pgvectorsc");
}
