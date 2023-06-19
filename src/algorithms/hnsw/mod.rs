mod implementation;
mod semaphore;
mod visited;

use super::Algorithm0;
use super::Algorithm1;
use crate::memory::ContextPtr;
use crate::prelude::BincodeDeserialize;
use crate::prelude::BincodeSerialize;
use crate::prelude::Options;
use crate::prelude::Scalar;
use implementation::Implementation;
use crate::memory::Address;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswOptions {
    #[serde(default = "HnswOptions::default_capacity")]
    pub capacity: usize,
    #[serde(default = "HnswOptions::default_build_threads")]
    pub build_threads: usize,
    #[serde(default = "HnswOptions::default_max_threads")]
    pub max_threads: usize,
    #[serde(default = "HnswOptions::default_m")]
    pub m: usize,
    #[serde(default = "HnswOptions::default_ef_construction")]
    pub ef_construction: usize,
    #[serde(default = "HnswOptions::default_max_level")]
    pub max_level: usize,
}

impl HnswOptions {
    fn default_capacity() -> usize {
        1 << 21
    }
    fn default_build_threads() -> usize {
        std::thread::available_parallelism().unwrap().get()
    }
    fn default_max_threads() -> usize {
        std::thread::available_parallelism().unwrap().get() * 2
    }
    fn default_m() -> usize {
        36
    }
    fn default_ef_construction() -> usize {
        500
    }
    fn default_max_level() -> usize {
        63
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswForever {
    options: Options,
    blocks: Vec<usize>,
    address: Address,
}

pub struct Hnsw {
    implementation: Implementation,
}

impl Algorithm0 for Hnsw {
    const BLOCKS: usize = 2;
    fn build(
        options: Options,
        data: async_channel::Receiver<(Vec<Scalar>, u64)>,
        blocks: Vec<usize>,
        context: ContextPtr,
    ) -> anyhow::Result<(Self, Vec<u8>)> {
        let hnsw_options = serde_json::from_str::<HnswOptions>(&options.options_algorithm)?;
        let implementation = Implementation::new(
            options.dims,
            hnsw_options.capacity,
            options.distance,
            hnsw_options.max_threads,
            hnsw_options.m,
            hnsw_options.ef_construction,
            hnsw_options.max_level,
            blocks.clone(),
            context,
        )?;
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let (tx, rx) = crossbeam::channel::bounded::<(Vec<Scalar>, u64)>(65536);
            let mut handles = vec![];
            let mut spawn = {
                let handles = &mut handles;
                let implementation = &implementation;
                move || {
                    let rx = rx.clone();
                    let handle = scope.spawn(move || -> anyhow::Result<()> {
                        while let Ok(data) = rx.recv() {
                            implementation.insert(data)?;
                        }
                        anyhow::Result::Ok(())
                    });
                    handles.push(handle);
                }
            };
            let mut take = 1024usize;
            while let Ok(data) = data.recv_blocking() {
                implementation.insert(data)?;
                take -= 1;
                if take == 0 {
                    break;
                }
            }
            for _ in 0..hnsw_options.build_threads {
                spawn();
            }
            while let Ok(data) = data.recv_blocking() {
                tx.send(data)?;
            }
            drop(tx);
            for handle in handles {
                handle
                    .join()
                    .ok()
                    .ok_or(anyhow::anyhow!("Thread exited."))??;
            }
            anyhow::Result::Ok(())
        })?;
        let forever = HnswForever { options, blocks, address: implementation.root.address() };
        Ok((Self { implementation }, forever.serialize()?))
    }

    fn load(forever: Vec<u8>, context: ContextPtr) -> anyhow::Result<Self> {
        let HnswForever { options, blocks, address } = forever.deserialize()?;
        let hnsw_options = serde_json::from_str::<HnswOptions>(&options.options_algorithm)?;
        let implementation = Implementation::load(
            options.distance,
            options.dims,
            hnsw_options.capacity,
            hnsw_options.max_threads,
            hnsw_options.m,
            hnsw_options.ef_construction,
            hnsw_options.max_level,
            blocks,
            address,
            context,
        )?;
        Ok(Self { implementation })
    }
}

impl Algorithm1 for Hnsw {
    fn insert(&self, insert: (Vec<Scalar>, u64)) -> anyhow::Result<()> {
        self.implementation.insert(insert)
    }

    fn search(&self, search: (Vec<Scalar>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        self.implementation.search(search)
    }
}
