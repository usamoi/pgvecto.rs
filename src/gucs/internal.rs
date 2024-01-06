use pgrx::{GucContext, GucFlags, GucRegistry, GucSetting};

#[derive(Debug, Clone, Copy, pgrx::PostgresGucEnum)]
#[allow(non_camel_case_types)]
pub enum Transport {
    unix,
    mmap,
}

impl Transport {
    pub const fn default() -> Transport {
        Transport::mmap
    }
}

pub static INTERNAL_TRANSPORT: GucSetting<Transport> =
    GucSetting::<Transport>::new(Transport::default());

pub unsafe fn init() {
    GucRegistry::define_enum_guc(
        "vectors.internal_transport",
        "IPC implementation.",
        "https://docs.pgvecto.rs/usage/search.html",
        &INTERNAL_TRANSPORT,
        GucContext::Userset,
        GucFlags::default(),
    );
}
