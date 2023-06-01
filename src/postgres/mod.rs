pub mod datatype;
mod index_hnsw;

pub type Id = pgrx::pg_sys::Oid;
pub type Scalar = f64;

#[repr(C, align(8))]
#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    Hash,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct HeapPointer {
    pub newtype: u64,
}

impl HeapPointer {
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
    pub fn into_u48(self) -> u64 {
        self.newtype
    }
}
