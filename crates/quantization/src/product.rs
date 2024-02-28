use crate::global::GlobalQuantization;
pub use base::global::*;
pub use base::index::*;
pub use base::scalar::*;
pub use base::vector::*;
use elkan_k_means::ElkanKMeans;
use num_traits::{Float, Zero};
use rand::seq::index::sample;
use rand::thread_rng;
use std::marker::PhantomData;
use std::path::Path;
use utils::mmap_array::MmapArray;
use utils::vec2::Vec2;

use super::QuantizationInput;

pub struct ProductQuantization<S: GlobalQuantization, I: QuantizationInput<S>> {
    dims: u16,
    ratio: u16,
    centroids: Vec<Scalar<S>>,
    codes: MmapArray<u8>,
    phantom: PhantomData<fn(I) -> I>,
}

unsafe impl<S: GlobalQuantization, I: QuantizationInput<S>> Send for ProductQuantization<S, I> {}
unsafe impl<S: GlobalQuantization, I: QuantizationInput<S>> Sync for ProductQuantization<S, I> {}

impl<S: GlobalQuantization, I: QuantizationInput<S>> ProductQuantization<S, I> {
    pub fn codes(&self, i: u32) -> &[u8] {
        let width = self.dims.div_ceil(self.ratio);
        let s = i as usize * width as usize;
        let e = (i + 1) as usize * width as usize;
        &self.codes[s..e]
    }
    pub fn create(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        raw: &I,
        permutation: Vec<u32>, // permutation is the mapping from placements to original ids
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let QuantizationOptions::Product(quantization_options) = quantization_options else {
            unreachable!()
        };
        let dims = options.vector.dims;
        let ratio = quantization_options.ratio as u16;
        let n = raw.len();
        let m = std::cmp::min(n, quantization_options.sample);
        let samples = {
            let f = sample(&mut thread_rng(), n as usize, m as usize).into_vec();
            let mut samples = Vec2::<Scalar<S>>::new(options.vector.dims, m as usize);
            for i in 0..m {
                samples[i as usize]
                    .copy_from_slice(raw.vector(f[i as usize] as u32).to_vec().as_ref());
            }
            samples
        };
        let width = dims.div_ceil(ratio);
        let mut centroids = vec![Scalar::<S>::zero(); 256 * dims as usize];
        for i in 0..width {
            let subdims = std::cmp::min(ratio, dims - ratio * i);
            let mut subsamples = Vec2::<Scalar<S>>::new(subdims, m as usize);
            for j in 0..m {
                let src = &samples[j as usize][(i * ratio) as usize..][..subdims as usize];
                subsamples[j as usize].copy_from_slice(src);
            }
            let mut k_means = ElkanKMeans::<S::ProductQuantizationL2>::new(256, subsamples);
            for _ in 0..25 {
                if k_means.iterate() {
                    break;
                }
            }
            let centroid = k_means.finish();
            for j in 0u8..=255 {
                centroids[j as usize * dims as usize..][(i * ratio) as usize..][..subdims as usize]
                    .copy_from_slice(&centroid[j as usize]);
            }
        }
        let codes_iter = (0..n).flat_map(|i| {
            let vector = raw.vector(permutation[i as usize]).to_vec();
            let width = dims.div_ceil(ratio);
            let mut result = Vec::with_capacity(width as usize);
            for i in 0..width {
                let subdims = std::cmp::min(ratio, dims - ratio * i);
                let mut minimal = F32::infinity();
                let mut target = 0u8;
                let left = &vector[(i * ratio) as usize..][..subdims as usize];
                for j in 0u8..=255 {
                    let right = &centroids[j as usize * dims as usize..][(i * ratio) as usize..]
                        [..subdims as usize];
                    let dis = S::product_quantization_l2_distance(left, right);
                    if dis < minimal {
                        minimal = dis;
                        target = j;
                    }
                }
                result.push(target);
            }
            result.into_iter()
        });
        std::fs::write(
            path.join("centroids"),
            serde_json::to_string(&centroids).unwrap(),
        )
        .unwrap();
        let codes = MmapArray::create(&path.join("codes"), codes_iter);
        utils::dir_ops::sync_dir(path);
        Self {
            dims,
            ratio,
            centroids,
            codes,
            phantom: PhantomData,
        }
    }

    pub fn open(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        _: &I,
    ) -> Self {
        let QuantizationOptions::Product(quantization_options) = quantization_options else {
            unreachable!()
        };
        let centroids =
            serde_json::from_slice(&std::fs::read(path.join("centroids")).unwrap()).unwrap();
        let codes = MmapArray::open(&path.join("codes"));
        Self {
            dims: options.vector.dims,
            ratio: quantization_options.ratio as _,
            centroids,
            codes,
            phantom: PhantomData,
        }
    }

    pub fn distance(&self, lhs: Borrowed<'_, S>, rhs: u32) -> F32 {
        let dims = self.dims;
        let ratio = self.ratio;
        let rhs = self.codes(rhs);
        S::product_quantization_distance(dims, ratio, &self.centroids, lhs, rhs)
    }

    pub fn distance2(&self, lhs: u32, rhs: u32) -> F32 {
        let dims = self.dims;
        let ratio = self.ratio;
        let lhs = self.codes(lhs);
        let rhs = self.codes(rhs);
        S::product_quantization_distance2(dims, ratio, &self.centroids, lhs, rhs)
    }
}
