//! Postgres vector extension.
//!
//! Provides an easy-to-use extension for vector similarity search.
#![feature(core_intrinsics)]
#![feature(allocator_api)]
#![feature(thread_local)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![feature(ptr_metadata)]
#![feature(new_uninit)]
#![feature(int_roundings)]
#![feature(never_type)]
#![feature(const_maybe_uninit_zeroed)]
#![feature(addr_parse_ascii)]
#![allow(clippy::complexity)]
#![allow(clippy::style)]

mod embedding;
mod postgres;

pgrx::pg_module_magic!();

pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[allow(non_snake_case)]
#[pgrx::pg_guard]
pub unsafe extern "C" fn _PG_init() {
    use pgrx::bgworkers::BackgroundWorkerBuilder;
    use pgrx::bgworkers::BgWorkerStartTime;
    #[cfg(debug_assertions)]
    {
        std::env::set_var("RUST_LIB_BACKTRACE", "1");
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    self::postgres::init();
    if self::postgres::gucs::REMOTE.get().is_none() {
        BackgroundWorkerBuilder::new("vectors")
            .set_function("bgworker")
            .set_library("vectors")
            .set_argument(None)
            .enable_shmem_access(None)
            .set_start_time(BgWorkerStartTime::PostmasterStart)
            .load();
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
