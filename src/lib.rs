//! Postgres vector extension.
//!
//! Provides an easy-to-use extension for vector similarity search.
#![feature(core_intrinsics)]
#![feature(allocator_api)]
#![feature(arbitrary_self_types)]
#![feature(panic_update_hook)]
#![feature(ptr_metadata)]
#![feature(lazy_cell)]
#![feature(try_blocks)]
#![feature(panic_info_message)]

use pgrx::prelude::*;

mod bgworker;
mod bridge;
mod embedding;
mod gucs;
mod operator;
mod postgres;
mod udf;

pgrx::pg_module_magic!();

pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[allow(non_snake_case)]
#[pg_guard]
pub unsafe extern "C" fn _PG_init() {
    use pgrx::bgworkers::BackgroundWorkerBuilder;
    use pgrx::bgworkers::BgWorkerStartTime;
    BackgroundWorkerBuilder::new("pgvectors")
        .set_function("pgvectors_main")
        .set_library("vectors")
        .set_argument(None)
        .set_start_time(BgWorkerStartTime::ConsistentState)
        .enable_spi_access()
        .enable_shmem_access(None)
        .load();
    gucs::init();
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
