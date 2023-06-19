use serde::{Deserialize, Serialize};

pub type Scalar = f32;

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum Error {
    #[error("The index is broken.")]
    IndexIsBroken,
    #[error("The index is not loaded.")]
    IndexIsUnloaded,
    #[error("Build an index with an invaild option.")]
    BuildOptionIsInvaild,
}

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
    #[allow(dead_code)]
    pub fn into_sys(self) -> pgrx::pg_sys::Oid {
        unsafe { pgrx::pg_sys::Oid::from_u32_unchecked(self.newtype) }
    }
    #[allow(dead_code)]
    pub fn from_u32(value: u32) -> Self {
        Self { newtype: value }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    pub dims: u16,
    pub distance: Distance,
    pub algorithm: String,
    pub options_algorithm: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Distance {
    L2,
    Cosine,
    Dot,
}

impl Distance {
    pub fn distance(self, lhs: &[Scalar], rhs: &[Scalar]) -> Scalar {
        if lhs.len() != rhs.len() {
            return Scalar::NAN;
        }
        let n = lhs.len();
        match self {
            Distance::L2 => {
                let mut result = 0.0 as Scalar;
                for i in 0..n {
                    result += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
                }
                result
            }
            Distance::Cosine => {
                let mut dot = 0.0 as Scalar;
                let mut x2 = 0.0 as Scalar;
                let mut y2 = 0.0 as Scalar;
                for i in 0..n {
                    dot += lhs[i] * rhs[i];
                    x2 += lhs[i] * lhs[i];
                    y2 += rhs[i] * rhs[i];
                }
                1.0 - dot * dot / (x2 * y2)
            }
            Distance::Dot => {
                let mut dot = 0.0 as Scalar;
                for i in 0..n {
                    dot += lhs[i] * rhs[i];
                }
                1.0 - dot
            }
        }
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
