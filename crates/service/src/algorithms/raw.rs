use crate::index::segments::growing::GrowingSegment;
use crate::index::segments::sealed::SealedSegment;
use crate::prelude::*;
use crate::storage::Storage;
use quantization::QuantizationInput;
use std::path::Path;
use std::sync::Arc;

pub struct Raw<S: G> {
    mmap: S::Storage,
}

impl<S: G> Raw<S> {
    pub fn create(
        path: &Path,
        options: IndexOptions,
        sealed: Vec<Arc<SealedSegment<S>>>,
        growing: Vec<Arc<GrowingSegment<S>>>,
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let ram = make(sealed, growing, options);
        let mmap = S::Storage::save(path, ram);
        crate::utils::dir_ops::sync_dir(path);
        Self { mmap }
    }

    pub fn open(path: &Path, options: IndexOptions) -> Self {
        Self {
            mmap: S::Storage::open(path, options),
        }
    }
}

impl<S: G> Raw<S> {
    pub fn len(&self) -> u32 {
        self.mmap.len()
    }

    pub fn vector(&self, i: u32) -> Borrowed<'_, S> {
        self.mmap.vector(i)
    }

    pub fn payload(&self, i: u32) -> Payload {
        self.mmap.payload(i)
    }
}

unsafe impl<S: G> Send for Raw<S> {}
unsafe impl<S: G> Sync for Raw<S> {}

pub struct RawRam<S: G> {
    sealed: Vec<Arc<SealedSegment<S>>>,
    growing: Vec<Arc<GrowingSegment<S>>>,
    dims: u16,
}

impl<S: G> RawRam<S> {
    pub fn dims(&self) -> u16 {
        self.dims
    }

    pub fn len(&self) -> u32 {
        self.sealed.iter().map(|x| x.len()).sum::<u32>()
            + self.growing.iter().map(|x| x.len()).sum::<u32>()
    }

    pub fn vector(&self, mut index: u32) -> Borrowed<'_, S> {
        for x in self.sealed.iter() {
            if index < x.len() {
                return x.vector(index);
            }
            index -= x.len();
        }
        for x in self.growing.iter() {
            if index < x.len() {
                return x.vector(index);
            }
            index -= x.len();
        }
        panic!("Out of bound.")
    }

    pub fn payload(&self, mut index: u32) -> Payload {
        for x in self.sealed.iter() {
            if index < x.len() {
                return x.payload(index);
            }
            index -= x.len();
        }
        for x in self.growing.iter() {
            if index < x.len() {
                return x.payload(index);
            }
            index -= x.len();
        }
        panic!("Out of bound.")
    }
}

fn make<S: G>(
    sealed: Vec<Arc<SealedSegment<S>>>,
    growing: Vec<Arc<GrowingSegment<S>>>,
    options: IndexOptions,
) -> RawRam<S> {
    RawRam {
        sealed,
        growing,
        dims: options.vector.dims,
    }
}

#[derive(Clone)]
pub struct ArcRaw<S: G>(pub Arc<Raw<S>>);

impl<S: G> QuantizationInput<S> for ArcRaw<S> {
    fn len(&self) -> u32 {
        Raw::len(&self.0)
    }

    fn vector(&self, i: u32) -> Borrowed<'_, S> {
        Raw::vector(&self.0, i)
    }

    fn payload(&self, i: u32) -> Payload {
        Raw::payload(&self.0, i)
    }
}
