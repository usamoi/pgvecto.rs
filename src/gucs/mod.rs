pub mod internal;
pub mod planning;
pub mod searching;

use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CStr;

// Not documented

pub static ENABLE_PREFILTER: GucSetting<bool> = GucSetting::<bool>::new(false);

pub static OPENAI_API_KEY_GUC: GucSetting<Option<&'static CStr>> =
    GucSetting::<Option<&'static CStr>>::new(None);

pub unsafe fn init() {
    unsafe {
        self::planning::init();
        self::searching::init();
        self::internal::init();
    }
    // Not documented
    GucRegistry::define_bool_guc(
        "vectors.enable_prefilter",
        "",
        "https://docs.pgvecto.rs/usage/search.html",
        &ENABLE_PREFILTER,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        "vectors.openai_api_key",
        "",
        "https://docs.pgvecto.rs/usage/search.html",
        &OPENAI_API_KEY_GUC,
        GucContext::Userset,
        GucFlags::default(),
    );
}
