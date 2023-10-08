use service::bgworker::WorkerOptions;
use service::ipc::transport::Address;

#[no_mangle]
extern "C" fn bgworker(_arg: pgrx::pg_sys::Datum) -> ! {
    match std::panic::catch_unwind(|| {
        std::fs::create_dir_all("pg_vectors").expect("Failed to create the directory.");
        service::bgworker::main(WorkerOptions {
            addr: Address::Unix("_socket".into()),
            chdir: "pg_vectors".into(),
        });
    }) {
        Ok(never) => never,
        Err(_) => {
            log::error!("The background process crashed.");
            pgrx::PANIC!("The background process crashed.");
        }
    }
}
