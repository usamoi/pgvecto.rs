use crate::algorithms::Vectors;
use crate::memory::using;
use crate::memory::Address;
use crate::prelude::Algorithm;
use crate::prelude::Backend;
use crate::prelude::BincodeDeserialize;
use crate::prelude::BincodeSerialize;
use crate::prelude::Options;
use crate::prelude::Scalar;
use crate::utils::hnsw::HnswImpl;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswOptions {
    pub backend: Backend,
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
pub struct HnswPersistent {
    address: Address,
}

pub struct Hnsw {
    implementation: HnswImpl,
}

impl Algorithm for Hnsw {
    type Options = HnswOptions;
    fn build(options: Options, vectors: Arc<Vectors>, n: usize) -> anyhow::Result<(Self, Vec<u8>)> {
        let hnsw_options = options.algorithm.clone().unwrap_hnsw();
        let implementation = HnswImpl::new(
            vectors.clone(),
            options.dims,
            options.distance,
            options.capacity,
            hnsw_options.max_threads,
            hnsw_options.m,
            hnsw_options.ef_construction,
            hnsw_options.max_level,
            using().allocator(hnsw_options.backend),
        )?;
        log::error!("n = {}", n);
        let i = AtomicUsize::new(0);
        using().scope(|scope| -> anyhow::Result<()> {
            let mut handles = Vec::new();
            for _ in 0..hnsw_options.build_threads {
                handles.push(scope.spawn(|| -> anyhow::Result<()> {
                    loop {
                        let i = i.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        implementation.insert(i)?;
                    }
                    anyhow::Result::Ok(())
                }));
            }
            for handle in handles.into_iter() {
                handle.join().unwrap()?;
            }
            anyhow::Result::Ok(())
        })?;
        let persistent = HnswPersistent {
            address: implementation.address,
        };
        Ok((Self { implementation }, persistent.serialize()?))
    }
    fn load(options: Options, vectors: Arc<Vectors>, persistent: Vec<u8>) -> anyhow::Result<Self> {
        let HnswPersistent { address } = persistent.deserialize()?;
        let hnsw_options = options.algorithm.unwrap_hnsw();
        let implementation = HnswImpl::load(
            vectors.clone(),
            options.distance,
            options.dims,
            options.capacity,
            hnsw_options.max_threads,
            hnsw_options.m,
            hnsw_options.ef_construction,
            hnsw_options.max_level,
            address,
            using().allocator(hnsw_options.backend),
        )?;
        Ok(Self { implementation })
    }
    fn insert(&self, insert: usize) -> anyhow::Result<()> {
        self.implementation.insert(insert)
    }
    fn search(&self, search: (Box<[Scalar]>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        self.implementation.search(search)
    }
}
