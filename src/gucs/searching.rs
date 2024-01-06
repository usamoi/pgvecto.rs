use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

// `search` searching mode

pub static SEARCH_MAXIMUM: GucSetting<i32> = GucSetting::<i32>::new(1024);

// IVF indexes

pub static IVF_NPROBE: GucSetting<i32> = GucSetting::<i32>::new(10);

// HNSW indexes

pub static HNSW_EF: GucSetting<i32> = GucSetting::<i32>::new(64);

pub unsafe fn init() {
    GucRegistry::define_int_guc(
        "vectors.search_maximum",
        "Maximum number of rows returned by a vector index.",
        "https://docs.pgvecto.rs/usage/search.html",
        &SEARCH_MAXIMUM,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.ivf_nporbe",
        "Number of lists to scan.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_NPROBE,
        1,
        1_000_000,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.hnsw_ef",
        "Search scope of HNSW.",
        "https://docs.pgvecto.rs/usage/search.html",
        &HNSW_EF,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
}
