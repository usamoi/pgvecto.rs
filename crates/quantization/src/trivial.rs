use super::QuantizationInput;
use crate::global::GlobalQuantization;
pub use base::global::*;
pub use base::index::*;
pub use base::scalar::*;
use std::marker::PhantomData;
use std::path::Path;

pub struct TrivialQuantization<S: GlobalQuantization, I: QuantizationInput<S>> {
    raw: I,
    permutation: Vec<u32>,
    phantom: PhantomData<fn(S) -> S>,
}

impl<S: GlobalQuantization, I: QuantizationInput<S>> TrivialQuantization<S, I> {
    pub fn codes(&self, i: u32) -> Borrowed<'_, S> {
        self.raw.vector(self.permutation[i as usize])
    }
    // permutation is the mapping from placements to original ids
    pub fn create(
        path: &Path,
        _: IndexOptions,
        _: QuantizationOptions,
        raw: &I,
        permutation: Vec<u32>,
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        // here we cannot modify raw, so we record permutation for translation
        std::fs::write(
            path.join("permutation"),
            serde_json::to_string(&permutation).unwrap(),
        )
        .unwrap();
        utils::dir_ops::sync_dir(path);
        Self {
            raw: raw.clone(),
            permutation,
            phantom: PhantomData,
        }
    }

    pub fn open(path: &Path, _: IndexOptions, _: QuantizationOptions, raw: &I) -> Self {
        let permutation =
            serde_json::from_slice(&std::fs::read(path.join("permutation")).unwrap()).unwrap();
        Self {
            raw: raw.clone(),
            permutation,
            phantom: PhantomData,
        }
    }

    pub fn distance(&self, lhs: Borrowed<'_, S>, rhs: u32) -> F32 {
        S::distance(lhs, self.codes(rhs))
    }

    pub fn distance2(&self, lhs: u32, rhs: u32) -> F32 {
        S::distance(self.codes(lhs), self.codes(rhs))
    }
}
