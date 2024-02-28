use super::QuantizationInput;
use crate::global::GlobalQuantization;
pub use base::global::*;
pub use base::index::*;
pub use base::scalar::*;
pub use base::vector::*;
use num_traits::Float;
use std::marker::PhantomData;
use std::path::Path;
use utils::mmap_array::MmapArray;

pub struct ScalarQuantization<S: GlobalQuantization, I: QuantizationInput<S>> {
    dims: u16,
    max: Vec<Scalar<S>>,
    min: Vec<Scalar<S>>,
    codes: MmapArray<u8>,
    phantom: PhantomData<fn(I) -> I>,
}

unsafe impl<S: GlobalQuantization, I: QuantizationInput<S>> Send for ScalarQuantization<S, I> {}
unsafe impl<S: GlobalQuantization, I: QuantizationInput<S>> Sync for ScalarQuantization<S, I> {}

impl<S: GlobalQuantization, I: QuantizationInput<S>> ScalarQuantization<S, I> {
    pub fn codes(&self, i: u32) -> &[u8] {
        let s = i as usize * self.dims as usize;
        let e = (i + 1) as usize * self.dims as usize;
        &self.codes[s..e]
    }
    pub fn create(
        path: &Path,
        options: IndexOptions,
        _: QuantizationOptions,
        raw: &I,
        permutation: Vec<u32>, // permutation is the mapping from placements to original ids
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let dims = options.vector.dims;
        let mut max = vec![Scalar::<S>::neg_infinity(); dims as usize];
        let mut min = vec![Scalar::<S>::infinity(); dims as usize];
        let n = raw.len();
        for i in 0..n {
            let vector = raw.vector(permutation[i as usize]).to_vec();
            for j in 0..dims as usize {
                max[j] = std::cmp::max(max[j], vector[j]);
                min[j] = std::cmp::min(min[j], vector[j]);
            }
        }
        std::fs::write(path.join("max"), serde_json::to_string(&max).unwrap()).unwrap();
        std::fs::write(path.join("min"), serde_json::to_string(&min).unwrap()).unwrap();
        let codes_iter = (0..n).flat_map(|i| {
            let vector = raw.vector(permutation[i as usize]).to_vec();
            let mut result = vec![0u8; dims as usize];
            for i in 0..dims as usize {
                let w = (((vector[i] - min[i]) / (max[i] - min[i])).to_f32() * 256.0) as u32;
                result[i] = w.clamp(0, 255) as u8;
            }
            result.into_iter()
        });
        let codes = MmapArray::create(&path.join("codes"), codes_iter);
        utils::dir_ops::sync_dir(path);
        Self {
            dims,
            max,
            min,
            codes,
            phantom: PhantomData,
        }
    }

    pub fn open(path: &Path, options: IndexOptions, _: QuantizationOptions, _: &I) -> Self {
        let dims = options.vector.dims;
        let max = serde_json::from_slice(&std::fs::read("max").unwrap()).unwrap();
        let min = serde_json::from_slice(&std::fs::read("min").unwrap()).unwrap();
        let codes = MmapArray::open(&path.join("codes"));
        Self {
            dims,
            max,
            min,
            codes,
            phantom: PhantomData,
        }
    }

    pub fn distance(&self, lhs: Borrowed<'_, S>, rhs: u32) -> F32 {
        let dims = self.dims;
        let rhs = self.codes(rhs);
        S::scalar_quantization_distance(dims, &self.max, &self.min, lhs, rhs)
    }

    pub fn distance2(&self, lhs: u32, rhs: u32) -> F32 {
        let dims = self.dims;
        let lhs = self.codes(lhs);
        let rhs = self.codes(rhs);
        S::scalar_quantization_distance2(dims, &self.max, &self.min, lhs, rhs)
    }
}
