#![feature(avx512_target_feature)]
#![allow(clippy::len_without_is_empty)]

pub mod global;

pub mod product;
pub mod scalar;
pub mod trivial;

use self::global::GlobalQuantization;
use self::product::ProductQuantization;
use self::scalar::ScalarQuantization;
use self::trivial::TrivialQuantization;

pub use base::global::*;
pub use base::index::*;
pub use base::scalar::*;
pub use base::search::*;
use std::path::Path;

pub trait QuantizationInput<S: GlobalQuantization>: Clone {
    fn len(&self) -> u32;

    fn vector(&self, i: u32) -> Borrowed<'_, S>;

    fn payload(&self, i: u32) -> Payload;
}

pub enum Quantization<S: GlobalQuantization, I: QuantizationInput<S>> {
    Trivial(TrivialQuantization<S, I>),
    Scalar(ScalarQuantization<S, I>),
    Product(ProductQuantization<S, I>),
}

impl<S: GlobalQuantization, I: QuantizationInput<S>> Quantization<S, I> {
    pub fn create(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        raw: &I,
        permutation: Vec<u32>, // permutation is the mapping from placements to original ids
    ) -> Self {
        match quantization_options {
            QuantizationOptions::Trivial(_) => Self::Trivial(TrivialQuantization::create(
                path,
                options,
                quantization_options,
                raw,
                permutation,
            )),
            QuantizationOptions::Scalar(_) => Self::Scalar(ScalarQuantization::create(
                path,
                options,
                quantization_options,
                raw,
                permutation,
            )),
            QuantizationOptions::Product(_) => Self::Product(ProductQuantization::create(
                path,
                options,
                quantization_options,
                raw,
                permutation,
            )),
        }
    }

    pub fn open(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        raw: &I,
    ) -> Self {
        match quantization_options {
            QuantizationOptions::Trivial(_) => Self::Trivial(TrivialQuantization::open(
                path,
                options,
                quantization_options,
                raw,
            )),
            QuantizationOptions::Scalar(_) => Self::Scalar(ScalarQuantization::open(
                path,
                options,
                quantization_options,
                raw,
            )),
            QuantizationOptions::Product(_) => Self::Product(ProductQuantization::open(
                path,
                options,
                quantization_options,
                raw,
            )),
        }
    }

    pub fn distance(&self, lhs: Borrowed<'_, S>, rhs: u32) -> F32 {
        use Quantization::*;
        match self {
            Trivial(x) => x.distance(lhs, rhs),
            Scalar(x) => x.distance(lhs, rhs),
            Product(x) => x.distance(lhs, rhs),
        }
    }

    pub fn distance2(&self, lhs: u32, rhs: u32) -> F32 {
        use Quantization::*;
        match self {
            Trivial(x) => x.distance2(lhs, rhs),
            Scalar(x) => x.distance2(lhs, rhs),
            Product(x) => x.distance2(lhs, rhs),
        }
    }
}
