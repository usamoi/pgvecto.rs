use base::global::*;
use base::scalar::F32;
use base::scalar::*;
use base::vector::*;
use num_traits::Float;

pub trait GlobalElkanKMeans: Global {
    fn elkan_k_means_normalize(vector: &mut [Scalar<Self>]);
    fn elkan_k_means_normalize2(vector: &mut Self::VectorOwned);
    fn elkan_k_means_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32;
    fn elkan_k_means_distance2(lhs: Borrowed<'_, Self>, rhs: &[Scalar<Self>]) -> F32;
}

impl GlobalElkanKMeans for SVecf32Cos {
    fn elkan_k_means_normalize(vector: &mut [Scalar<Self>]) {
        base::computation::vecf32::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut SVecf32Owned) {
        base::computation::svecf32::l2_normalize(vector)
    }

    fn elkan_k_means_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32 {
        base::computation::vecf32::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Borrowed<'_, Self>, rhs: &[Scalar<Self>]) -> F32 {
        base::computation::svecf32::dot_2(lhs, rhs).acos()
    }
}

impl GlobalElkanKMeans for SVecf32Dot {
    fn elkan_k_means_normalize(vector: &mut [Scalar<Self>]) {
        base::computation::vecf32::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut SVecf32Owned) {
        base::computation::svecf32::l2_normalize(vector)
    }

    fn elkan_k_means_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32 {
        base::computation::vecf32::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Borrowed<'_, Self>, rhs: &[Scalar<Self>]) -> F32 {
        base::computation::svecf32::dot_2(lhs, rhs).acos()
    }
}

impl GlobalElkanKMeans for SVecf32L2 {
    fn elkan_k_means_normalize(_: &mut [Scalar<Self>]) {}

    fn elkan_k_means_normalize2(_: &mut SVecf32Owned) {}

    fn elkan_k_means_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32 {
        base::computation::vecf32::sl2(lhs, rhs).sqrt()
    }

    fn elkan_k_means_distance2(lhs: SVecf32Borrowed<'_>, rhs: &[Scalar<Self>]) -> F32 {
        base::computation::svecf32::sl2_2(lhs, rhs).sqrt()
    }
}

impl GlobalElkanKMeans for Vecf16Cos {
    fn elkan_k_means_normalize(vector: &mut [F16]) {
        base::computation::vecf16::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut Vecf16Owned) {
        base::computation::vecf16::l2_normalize(vector.slice_mut())
    }

    fn elkan_k_means_distance(lhs: &[F16], rhs: &[F16]) -> F32 {
        base::computation::vecf16::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Vecf16Borrowed<'_>, rhs: &[F16]) -> F32 {
        base::computation::vecf16::dot(lhs.slice(), rhs).acos()
    }
}

impl GlobalElkanKMeans for Vecf16Dot {
    fn elkan_k_means_normalize(vector: &mut [F16]) {
        base::computation::vecf16::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut Vecf16Owned) {
        base::computation::vecf16::l2_normalize(vector.slice_mut())
    }

    fn elkan_k_means_distance(lhs: &[F16], rhs: &[F16]) -> F32 {
        base::computation::vecf16::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Vecf16Borrowed<'_>, rhs: &[F16]) -> F32 {
        base::computation::vecf16::dot(lhs.slice(), rhs).acos()
    }
}

impl GlobalElkanKMeans for Vecf16L2 {
    fn elkan_k_means_normalize(_: &mut [F16]) {}

    fn elkan_k_means_normalize2(_: &mut Vecf16Owned) {}

    fn elkan_k_means_distance(lhs: &[F16], rhs: &[F16]) -> F32 {
        base::computation::vecf16::sl2(lhs, rhs).sqrt()
    }

    fn elkan_k_means_distance2(lhs: Vecf16Borrowed<'_>, rhs: &[F16]) -> F32 {
        base::computation::vecf16::sl2(lhs.slice(), rhs).sqrt()
    }
}

impl GlobalElkanKMeans for Vecf32Cos {
    fn elkan_k_means_normalize(vector: &mut [F32]) {
        base::computation::vecf32::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut Vecf32Owned) {
        base::computation::vecf32::l2_normalize(vector.slice_mut())
    }

    fn elkan_k_means_distance(lhs: &[F32], rhs: &[F32]) -> F32 {
        base::computation::vecf32::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Vecf32Borrowed<'_>, rhs: &[F32]) -> F32 {
        base::computation::vecf32::dot(lhs.slice(), rhs).acos()
    }
}

impl GlobalElkanKMeans for Vecf32Dot {
    fn elkan_k_means_normalize(vector: &mut [F32]) {
        base::computation::vecf32::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut Vecf32Owned) {
        base::computation::vecf32::l2_normalize(vector.slice_mut())
    }

    fn elkan_k_means_distance(lhs: &[F32], rhs: &[F32]) -> F32 {
        base::computation::vecf32::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Vecf32Borrowed<'_>, rhs: &[F32]) -> F32 {
        base::computation::vecf32::dot(lhs.slice(), rhs).acos()
    }
}

impl GlobalElkanKMeans for Vecf32L2 {
    fn elkan_k_means_normalize(_: &mut [F32]) {}

    fn elkan_k_means_normalize2(_: &mut Vecf32Owned) {}

    fn elkan_k_means_distance(lhs: &[Scalar<Self>], rhs: &[Scalar<Self>]) -> F32 {
        base::computation::vecf32::sl2(lhs, rhs).sqrt()
    }

    fn elkan_k_means_distance2(lhs: Vecf32Borrowed<'_>, rhs: &[Scalar<Self>]) -> F32 {
        base::computation::vecf32::sl2(lhs.slice(), rhs).sqrt()
    }
}
