use crate::algorithms::Vectors;
use crate::memory::using;
use crate::memory::Address;
use crate::prelude::Algorithm;
use crate::prelude::Backend;
use crate::prelude::BincodeDeserialize;
use crate::prelude::BincodeSerialize;
use crate::prelude::Options;
use crate::prelude::Scalar;
use crate::utils::ivf::IvfImpl;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IvfOptions {
    pub backend: Backend,
    #[serde(default = "IvfOptions::default_build_threads")]
    pub build_threads: usize,
    #[serde(default = "IvfOptions::default_nlist")]
    pub nlist: usize,
    #[serde(default = "IvfOptions::default_nsample")]
    pub nsample: usize,
    #[serde(default = "IvfOptions::default_nprobe")]
    pub nprobe: usize,
}

impl IvfOptions {
    fn default_build_threads() -> usize {
        std::thread::available_parallelism().unwrap().get()
    }
    fn default_nlist() -> usize {
        32
    }
    fn default_nsample() -> usize {
        2000
    }
    fn default_nprobe() -> usize {
        10
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IvfPersistent {
    address: Address,
}

pub struct Ivf {
    implementation: IvfImpl,
}

impl Algorithm for Ivf {
    type Options = IvfOptions;
    fn build(options: Options, vectors: Arc<Vectors>, n: usize) -> anyhow::Result<(Self, Vec<u8>)> {
        let ivf_options = options.algorithm.clone().unwrap_ivf();
        let implementation = IvfImpl::new(
            vectors.clone(),
            options.dims,
            options.distance,
            vectors.len(),
            ivf_options.nlist,
            ivf_options.nsample,
            ivf_options.nprobe,
            options.capacity,
            using().allocator(ivf_options.backend),
        )?;
        let i = AtomicUsize::new(0);
        using().scope(|scope| {
            for _ in 0..ivf_options.build_threads {
                scope.spawn(|| -> anyhow::Result<()> {
                    loop {
                        let i = i.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        implementation.insert(i)?;
                    }
                    anyhow::Result::Ok(())
                });
            }
        });
        let persistent = IvfPersistent {
            address: implementation.address,
        };
        Ok((Self { implementation }, persistent.serialize()?))
    }
    fn load(options: Options, vectors: Arc<Vectors>, persistent: Vec<u8>) -> anyhow::Result<Self> {
        let IvfPersistent { address } = persistent.deserialize()?;
        let implementation = IvfImpl::load(vectors.clone(), options.distance, address)?;
        Ok(Self { implementation })
    }
    fn insert(&self, insert: usize) -> anyhow::Result<()> {
        self.implementation.insert(insert)
    }
    fn search(&self, search: (Box<[Scalar]>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        self.implementation.search(search)
    }
}
