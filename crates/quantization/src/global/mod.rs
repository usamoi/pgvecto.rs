mod product;
mod scalar;

use base::global::*;
use base::scalar::F32;
use elkan_k_means::global::GlobalElkanKMeans;

pub trait GlobalScalarQuantization: Global {
    fn scalar_quantization_distance(
        dims: u16,
        max: &[Scalar<Self>],
        min: &[Scalar<Self>],
        lhs: Borrowed<'_, Self>,
        rhs: &[u8],
    ) -> F32;
    fn scalar_quantization_distance2(
        dims: u16,
        max: &[Scalar<Self>],
        min: &[Scalar<Self>],
        lhs: &[u8],
        rhs: &[u8],
    ) -> F32;
}

pub trait GlobalProductQuantization: Global {
    type ProductQuantizationL2: Global<VectorOwned = Self::VectorOwned>
        + GlobalElkanKMeans
        + GlobalProductQuantization;
    fn product_quantization_distance(
        dims: u16,
        ratio: u16,
        centroids: &[Scalar<Self>],
        lhs: Borrowed<'_, Self>,
        rhs: &[u8],
    ) -> F32;
    fn product_quantization_distance2(
        dims: u16,
        ratio: u16,
        centroids: &[Scalar<Self>],
        lhs: &[u8],
        rhs: &[u8],
    ) -> F32;
    fn product_quantization_distance_with_delta(
        dims: u16,
        ratio: u16,
        centroids: &[Scalar<Self>],
        lhs: Borrowed<'_, Self>,
        rhs: &[u8],
        delta: &[Scalar<Self>],
    ) -> F32;
    fn product_quantization_l2_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32;
    fn product_quantization_dense_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32;
}

pub trait GlobalQuantization:
    Global + GlobalScalarQuantization + GlobalProductQuantization
{
}

impl GlobalQuantization for SVecf32Cos {}
impl GlobalQuantization for SVecf32Dot {}
impl GlobalQuantization for SVecf32L2 {}
impl GlobalQuantization for Vecf32Cos {}
impl GlobalQuantization for Vecf32Dot {}
impl GlobalQuantization for Vecf32L2 {}
impl GlobalQuantization for Vecf16Cos {}
impl GlobalQuantization for Vecf16Dot {}
impl GlobalQuantization for Vecf16L2 {}
