#[no_mangle]
extern "C" fn bgworker(_arg: pgrx::pg_sys::Datum) -> ! {
    match std::panic::catch_unwind(service::bgworker::main) {
        Ok(never) => never,
        Err(_) => {
            log::error!("The background process crashed.");
            pgrx::PANIC!("The background process crashed.");
        }
    }
}
