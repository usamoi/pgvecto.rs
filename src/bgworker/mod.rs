mod algo_hnsw;
pub mod export;
mod index;
mod rpc;
mod wal;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Options {
    pub common: CommonOptions,
    pub algo: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CommonOptions {
    pub dims: u16,
}

use crate::bgworker::rpc::thread_session;
use crate::postgres::{HeapPointer, Id, Scalar};
use index::Index;
use pgrx::bgworkers::{BackgroundWorker, SignalWakeFlags};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, RwLock};

struct Global {
    indexes: Mutex<HashMap<Id, Option<Arc<RwLock<Index>>>>>,
}

static GLOBAL: MaybeUninit<Global> = MaybeUninit::uninit();

fn global() -> &'static Global {
    unsafe { GLOBAL.assume_init_ref() }
}

fn global_index_load(id: Id) -> Arc<RwLock<Index>> {
    loop {
        {
            let mut guard = global().indexes.lock().unwrap();
            if let Some(x) = guard.get_mut(&id) {
                if let Some(x) = x {
                    return x.clone();
                } else {
                    std::thread::yield_now();
                    continue;
                }
            } else {
                guard.insert(id, None);
            }
        }
        let index;
        {
            index = Arc::new(RwLock::new(Index::load(id)));
            let mut guard = global().indexes.lock().unwrap();
            *guard.get_mut(&id).unwrap() = Some(index.clone());
        }
        return index;
    }
}

fn global_index_build(
    id: Id,
    options: Options,
    parts: impl Iterator<Item = (Vec<Scalar>, HeapPointer)>,
) -> Arc<RwLock<Index>> {
    loop {
        {
            let mut guard = global().indexes.lock().unwrap();
            if let Some(x) = guard.get_mut(&id) {
                if let Some(x) = x {
                    return x.clone();
                } else {
                    std::thread::yield_now();
                    continue;
                }
            } else {
                guard.insert(id, None);
            }
        }
        let index;
        {
            index = Arc::new(RwLock::new(Index::build(id, options, parts)));
            let mut guard = global().indexes.lock().unwrap();
            *guard.get_mut(&id).unwrap() = Some(index.clone());
        }
        return index;
    }
}

#[no_mangle]
extern "C" fn pgvectors_main(_arg: pgrx::pg_sys::Datum) -> ! {
    match std::panic::catch_unwind(|| {
        main();
    }) {
        Ok(never) => never,
        Err(_) => pgrx::PANIC!("The background process crashed."),
    }
}

fn main() -> ! {
    std::fs::create_dir_all("pg_vectors").unwrap();
    std::env::set_current_dir("pg_vectors").unwrap();
    unsafe {
        let global = Global {
            indexes: Mutex::new(HashMap::new()),
        };
        (GLOBAL.as_ptr() as *mut Global).write(global);
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open("_log")
        .unwrap();
    env_logger::builder()
        .target(env_logger::Target::Pipe(Box::new(log)))
        .init();
    std::panic::update_hook(|prev, info| {
        log::error!("Panicked at {:?}. {:?}", info.location(), info.message());
        prev(info);
    });
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    let listener = std::net::TcpListener::bind("0.0.0.0:33509").unwrap();
    loop {
        let (stream, _) = listener.accept().unwrap();
        std::thread::spawn(move || thread_session(stream));
    }
}
