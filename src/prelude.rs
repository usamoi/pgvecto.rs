use std::sync::Arc;

use crate::algorithms::Flat;
use crate::algorithms::Hnsw;
use crate::algorithms::Ivf;
use crate::algorithms::Vectors;
use serde::{Deserialize, Serialize};

pub type Scalar = f32;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Id {
    pub newtype: u32,
}

impl Id {
    pub fn from_sys(sys: pgrx::pg_sys::Oid) -> Self {
        Self {
            newtype: sys.as_u32(),
        }
    }
    pub fn as_u32(self) -> u32 {
        self.newtype
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pointer {
    pub newtype: u64,
}

impl Pointer {
    pub fn from_sys(sys: pgrx::pg_sys::ItemPointerData) -> Self {
        let mut newtype = 0;
        newtype |= (sys.ip_blkid.bi_hi as u64) << 32;
        newtype |= (sys.ip_blkid.bi_lo as u64) << 16;
        newtype |= (sys.ip_posid as u64) << 0;
        Self { newtype }
    }
    pub fn into_sys(self) -> pgrx::pg_sys::ItemPointerData {
        pgrx::pg_sys::ItemPointerData {
            ip_blkid: pgrx::pg_sys::BlockIdData {
                bi_hi: ((self.newtype >> 32) & 0xffff) as u16,
                bi_lo: ((self.newtype >> 16) & 0xffff) as u16,
            },
            ip_posid: ((self.newtype >> 0) & 0xffff) as u16,
        }
    }
    pub fn from_u48(value: u64) -> Self {
        assert!(value < (1u64 << 48));
        Self { newtype: value }
    }
    pub fn as_u48(self) -> u64 {
        self.newtype
    }
}

pub trait Algorithm: Sized {
    type Options: Clone + serde::Serialize + serde::de::DeserializeOwned;
    fn build(options: Options, vectors: Arc<Vectors>, n: usize) -> anyhow::Result<(Self, Vec<u8>)>;
    fn load(options: Options, vectors: Arc<Vectors>, persistent: Vec<u8>) -> anyhow::Result<Self>;
    fn insert(&self, i: usize) -> anyhow::Result<()>;
    fn search(&self, search: (Box<[Scalar]>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    pub dims: u16,
    pub distance: Distance,
    pub capacity: usize,
    pub size_ram: usize,
    pub size_disk: usize,
    pub backend_vectors: Backend,
    pub algorithm: AlgorithmOptions,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum Backend {
    Ram = 0,
    Disk = 1,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AlgorithmOptions {
    Hnsw(<Hnsw as Algorithm>::Options),
    Flat(<Flat as Algorithm>::Options),
    Ivf(<Ivf as Algorithm>::Options),
}

impl AlgorithmOptions {
    pub fn name(&self) -> &str {
        use AlgorithmOptions::*;
        match self {
            Hnsw(_) => "Hnsw",
            Flat(_) => "Flat",
            Ivf(_) => "Ivf",
        }
    }
    pub fn unwrap_hnsw(self) -> <Hnsw as Algorithm>::Options {
        use AlgorithmOptions::*;
        match self {
            Hnsw(x) => x,
            _ => unreachable!(),
        }
    }
    #[allow(dead_code)]
    pub fn unwrap_flat(self) -> <Flat as Algorithm>::Options {
        use AlgorithmOptions::*;
        match self {
            Flat(x) => x,
            _ => unreachable!(),
        }
    }
    pub fn unwrap_ivf(self) -> <Ivf as Algorithm>::Options {
        use AlgorithmOptions::*;
        match self {
            Ivf(x) => x,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Distance {
    L2,
    Cosine,
    Dot,
}

impl Distance {
    pub fn distance(self, lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        match self {
            Distance::L2 => Self::l2(lhs, rhs),
            Distance::Cosine => Self::cosine(lhs, rhs),
            Distance::Dot => Self::dot(lhs, rhs),
        }
    }
    pub fn l2(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        use core::intrinsics::fadd_fast as add;
        use core::intrinsics::fmul_fast as mul;
        use core::intrinsics::fsub_fast as sub;
        if lhs.len() != rhs.len() {
            return Scalar::NAN;
        }
        let n = lhs.len();
        let mut result = 0.0 as Scalar;
        for i in 0..n {
            unsafe {
                result = add(result, mul(sub(lhs[i], rhs[i]), sub(lhs[i], rhs[i])));
            }
        }
        result
    }
    pub fn cosine(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        use core::intrinsics::fadd_fast as add;
        use core::intrinsics::fmul_fast as mul;
        if lhs.len() != rhs.len() {
            return Scalar::NAN;
        }
        let n = lhs.len();
        let mut dot = 0.0 as Scalar;
        let mut x2 = 0.0 as Scalar;
        let mut y2 = 0.0 as Scalar;
        for i in 0..n {
            unsafe {
                dot = add(dot, mul(lhs[i], rhs[i]));
                x2 = add(x2, mul(lhs[i], lhs[i]));
                y2 = add(y2, mul(rhs[i], rhs[i]));
            }
        }
        1.0 - dot * dot / (x2 * y2)
    }
    pub fn dot(lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        use core::intrinsics::fadd_fast as add;
        use core::intrinsics::fmul_fast as mul;
        if lhs.len() != rhs.len() {
            return Scalar::NAN;
        }
        let n = lhs.len();
        let mut dot = 0.0 as Scalar;
        for i in 0..n {
            unsafe {
                dot = add(dot, mul(lhs[i], rhs[i]));
            }
        }
        1.0 - dot
    }
}

pub trait BincodeDeserialize {
    fn deserialize<'a, T: Deserialize<'a>>(&'a self) -> anyhow::Result<T>;
}

impl BincodeDeserialize for [u8] {
    fn deserialize<'a, T: Deserialize<'a>>(&'a self) -> anyhow::Result<T> {
        let t = bincode::deserialize::<T>(self)?;
        Ok(t)
    }
}

pub trait BincodeSerialize {
    fn serialize(&self) -> anyhow::Result<Vec<u8>>;
}

impl<T: Serialize> BincodeSerialize for T {
    fn serialize(&self) -> anyhow::Result<Vec<u8>> {
        let bytes = bincode::serialize(self)?;
        Ok(bytes)
    }
}
