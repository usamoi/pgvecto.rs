mod hnsw;

use self::hnsw::Hnsw;
use crate::memory::ContextPtr;
use crate::prelude::*;

pub trait Algorithm0: Sized {
    const BLOCKS: usize;
    fn build(
        options: Options,
        data: async_channel::Receiver<(Vec<Scalar>, u64)>,
        blocks: Vec<usize>,
        context: ContextPtr,
    ) -> anyhow::Result<(Self, Vec<u8>)>;
    fn load(forever: Vec<u8>, context: ContextPtr) -> anyhow::Result<Self>;
}

pub trait Algorithm1 {
    fn insert(&self, insert: (Vec<Scalar>, u64)) -> anyhow::Result<()>;
    fn search(&self, search: (Vec<Scalar>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>>;
}

pub trait Algorithm: Algorithm0 + Algorithm1 {}

impl<T: Algorithm0 + Algorithm1> Algorithm for T {}

#[derive(Debug, Clone, Copy)]
pub enum Algo0 {
    Hnsw,
}

impl Algo0 {
    pub fn new(name: &str) -> anyhow::Result<Self> {
        match name {
            "HNSW" => Ok(Self::Hnsw),
            _ => anyhow::bail!(Error::BuildOptionIsInvaild),
        }
    }
    pub fn blocks(&self) -> usize {
        match self {
            Self::Hnsw => Hnsw::BLOCKS,
        }
    }
    pub async fn build(
        self,
        options: Options,
        data: async_channel::Receiver<(Vec<Scalar>, u64)>,
        blocks: Vec<usize>,
        context: ContextPtr,
    ) -> anyhow::Result<(Algo1, Vec<u8>)> {
        fn f<T, U: From<T>>((algo, forever): (T, Vec<u8>)) -> (U, Vec<u8>) {
            (algo.into(), forever)
        }
        tokio::task::block_in_place(|| {
            Ok(match self {
                Self::Hnsw => f(Hnsw::build(options, data, blocks, context)?),
            })
        })
    }
    pub async fn load(self, forever: Vec<u8>, context: ContextPtr) -> anyhow::Result<Algo1> {
        tokio::task::block_in_place(|| {
            Ok(match self {
                Self::Hnsw => Hnsw::load(forever, context)?.into(),
            })
        })
    }
}

pub enum Algo1 {
    Hnsw(Hnsw),
}

impl From<Hnsw> for Algo1 {
    fn from(value: Hnsw) -> Self {
        Self::Hnsw(value)
    }
}

impl Algo1 {
    pub async fn insert(&self, insert: (Vec<Scalar>, u64)) -> anyhow::Result<()> {
        use Algo1::*;
        tokio::task::block_in_place(|| match self {
            Hnsw(x) => x.insert(insert),
        })
    }
    pub async fn search(&self, search: (Vec<Scalar>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        use Algo1::*;
        tokio::task::block_in_place(|| match self {
            Hnsw(x) => x.search(search),
        })
    }
}
