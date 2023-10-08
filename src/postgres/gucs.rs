use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};
use service::ipc::transport::Address;
use std::{ffi::CStr, net::SocketAddr};

pub static OPENAI_API_KEY_GUC: GucSetting<Option<&'static CStr>> =
    GucSetting::<Option<&'static CStr>>::new(None);

pub static REMOTE: GucSetting<Option<&'static CStr>> =
    GucSetting::<Option<&'static CStr>>::new(None);

pub static K: GucSetting<i32> = GucSetting::<i32>::new(64);

pub unsafe fn init() {
    GucRegistry::define_string_guc(
        "vectors.openai_api_key",
        "The API key of OpenAI.",
        "The OpenAI API key is required to use OpenAI embedding.",
        &OPENAI_API_KEY_GUC,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.k",
        "The number of nearest neighbors to return for searching.",
        "The number of nearest neighbors to return for searching.",
        &K,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        "vectors.remote",
        "Remote daemon address to connect.",
        "Remote daemon address to connect. If not set, run a background process locally.",
        &REMOTE,
        GucContext::Postmaster,
        GucFlags::default(),
    );
}

pub fn addr() -> Address {
    match REMOTE.get() {
        Some(s) => Address::Tcp(SocketAddr::parse_ascii(s.to_bytes()).expect("Bad address.")),
        None => Address::Unix("./pg_vectors/_socket".into()),
    }
}
