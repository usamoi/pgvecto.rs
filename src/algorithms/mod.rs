mod flat;
mod hnsw;
mod ivf;
mod vectors;

pub use flat::Flat;
pub use hnsw::Hnsw;
pub use ivf::Ivf;
pub use vectors::Vectors;

use crate::prelude::*;
use std::sync::Arc;

pub enum IndexAlgo {
    Hnsw(Hnsw),
    Flat(Flat),
    Ivf(Ivf),
}

impl IndexAlgo {
    pub fn build(
        name: &str,
        options: Options,
        vectors: Arc<Vectors>,
        n: usize,
    ) -> anyhow::Result<(Self, Vec<u8>)> {
        match name {
            "Hnsw" => Ok({
                let x = Hnsw::build(options, vectors, n)?;
                (Self::Hnsw(x.0), x.1)
            }),
            "Flat" => Ok({
                let x = Flat::build(options, vectors, n)?;
                (Self::Flat(x.0), x.1)
            }),
            "Ivf" => Ok({
                let x = Ivf::build(options, vectors, n)?;
                (Self::Ivf(x.0), x.1)
            }),
            _ => anyhow::bail!("Unknown algorithm."),
        }
    }
    pub fn load(
        name: &str,
        options: Options,
        vectors: Arc<Vectors>,
        persistent: Vec<u8>,
    ) -> anyhow::Result<IndexAlgo> {
        match name {
            "Hnsw" => Ok(Self::Hnsw(Hnsw::load(options, vectors, persistent)?)),
            "Flat" => Ok(Self::Flat(Flat::load(options, vectors, persistent)?)),
            _ => anyhow::bail!("Unknown algorithm."),
        }
    }
    pub fn insert(&self, insert: usize) -> anyhow::Result<()> {
        use IndexAlgo::*;
        match self {
            Hnsw(x) => x.insert(insert),
            Flat(x) => x.insert(insert),
            Ivf(x) => x.insert(insert),
        }
    }
    pub fn search(&self, search: (Box<[Scalar]>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        use IndexAlgo::*;
        match self {
            Hnsw(x) => x.search(search),
            Flat(x) => x.search(search),
            Ivf(x) => x.search(search),
        }
    }
}
