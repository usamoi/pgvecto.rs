use crate::distance::*;
use crate::scalar::*;
use crate::vector::*;

pub trait Global: Copy + 'static {
    type VectorOwned: VectorOwned;

    const VECTOR_KIND: VectorKind;
    const DISTANCE_KIND: DistanceKind;

    fn distance(lhs: Borrowed<'_, Self>, rhs: Borrowed<'_, Self>) -> F32;
}

pub type Owned<T> = <T as Global>::VectorOwned;
pub type Borrowed<'a, T> = <<T as Global>::VectorOwned as VectorOwned>::Borrowed<'a>;
pub type Scalar<T> = <<T as Global>::VectorOwned as VectorOwned>::Scalar;

#[derive(Debug, Clone, Copy)]
pub enum SVecf32Cos {}

impl Global for SVecf32Cos {
    type VectorOwned = SVecf32Owned;

    const VECTOR_KIND: VectorKind = VectorKind::SVecf32;
    const DISTANCE_KIND: DistanceKind = DistanceKind::Cos;

    fn distance(lhs: Borrowed<'_, Self>, rhs: Borrowed<'_, Self>) -> F32 {
        F32(1.0) - crate::computation::svecf32::cosine(lhs, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SVecf32Dot {}

impl Global for SVecf32Dot {
    type VectorOwned = SVecf32Owned;

    const VECTOR_KIND: VectorKind = VectorKind::SVecf32;
    const DISTANCE_KIND: DistanceKind = DistanceKind::Dot;

    fn distance(lhs: Borrowed<'_, Self>, rhs: Borrowed<'_, Self>) -> F32 {
        crate::computation::svecf32::dot(lhs, rhs) * (-1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SVecf32L2 {}

impl Global for SVecf32L2 {
    type VectorOwned = SVecf32Owned;

    const VECTOR_KIND: VectorKind = VectorKind::SVecf32;
    const DISTANCE_KIND: DistanceKind = DistanceKind::L2;

    fn distance(lhs: SVecf32Borrowed<'_>, rhs: SVecf32Borrowed<'_>) -> F32 {
        crate::computation::svecf32::sl2(lhs, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf16Cos {}

impl Global for Vecf16Cos {
    type VectorOwned = Vecf16Owned;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf16;
    const DISTANCE_KIND: DistanceKind = DistanceKind::Cos;

    fn distance(lhs: Vecf16Borrowed<'_>, rhs: Vecf16Borrowed<'_>) -> F32 {
        F32(1.0) - crate::computation::vecf16::cosine(lhs.slice(), rhs.slice())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf16Dot {}

impl Global for Vecf16Dot {
    type VectorOwned = Vecf16Owned;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf16;
    const DISTANCE_KIND: DistanceKind = DistanceKind::Dot;

    fn distance(lhs: Vecf16Borrowed<'_>, rhs: Vecf16Borrowed<'_>) -> F32 {
        crate::computation::vecf16::dot(lhs.slice(), rhs.slice()) * (-1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf16L2 {}

impl Global for Vecf16L2 {
    type VectorOwned = Vecf16Owned;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf16;
    const DISTANCE_KIND: DistanceKind = DistanceKind::L2;

    fn distance(lhs: Vecf16Borrowed<'_>, rhs: Vecf16Borrowed<'_>) -> F32 {
        crate::computation::vecf16::sl2(lhs.slice(), rhs.slice())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf32Cos {}

impl Global for Vecf32Cos {
    type VectorOwned = Vecf32Owned;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf32;
    const DISTANCE_KIND: DistanceKind = DistanceKind::Cos;

    fn distance(lhs: Vecf32Borrowed<'_>, rhs: Vecf32Borrowed<'_>) -> F32 {
        F32(1.0) - crate::computation::vecf32::cosine(lhs.slice(), rhs.slice())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf32Dot {}

impl Global for Vecf32Dot {
    type VectorOwned = Vecf32Owned;

    const DISTANCE_KIND: DistanceKind = DistanceKind::Dot;
    const VECTOR_KIND: VectorKind = VectorKind::Vecf32;

    fn distance(lhs: Vecf32Borrowed<'_>, rhs: Vecf32Borrowed<'_>) -> F32 {
        crate::computation::vecf32::dot(lhs.slice(), rhs.slice()) * (-1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vecf32L2 {}

impl Global for Vecf32L2 {
    type VectorOwned = Vecf32Owned;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf32;
    const DISTANCE_KIND: DistanceKind = DistanceKind::L2;

    fn distance(lhs: Vecf32Borrowed<'_>, rhs: Vecf32Borrowed<'_>) -> F32 {
        crate::computation::vecf32::sl2(lhs.slice(), rhs.slice())
    }
}
