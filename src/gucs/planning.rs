use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

pub static ENABLE_INDEX: GucSetting<bool> = GucSetting::<bool>::new(true);

pub static ENABLE_SEARCH: GucSetting<bool> = GucSetting::<bool>::new(true);

pub static ENABLE_VBASE: GucSetting<bool> = GucSetting::<bool>::new(false);

pub unsafe fn init() {
    GucRegistry::define_bool_guc(
        "vectors.enable_index",
        "Enable vector indexes.",
        "https://docs.pgvecto.rs/usage/search.html",
        &ENABLE_INDEX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.enable_search",
        "Enable `search` searching mode.",
        "https://docs.pgvecto.rs/usage/search.html",
        &ENABLE_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.enable_vbase",
        "Enable `vbase` searching mode.",
        "https://docs.pgvecto.rs/usage/search.html",
        &ENABLE_VBASE,
        GucContext::Userset,
        GucFlags::default(),
    );
}
