use crate::algorithms::raw::Raw;
use crate::index::segments::growing::GrowingSegment;
use crate::index::segments::sealed::SealedSegment;
use crate::prelude::*;
use elkan_k_means::ElkanKMeans;
use quantization::global::GlobalProductQuantization;
use rand::seq::index::sample;
use rand::thread_rng;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::Arc;
use utils::mmap_array::MmapArray;
use utils::vec2::Vec2;

pub struct IvfPq<S: G> {
    mmap: IvfMmap<S>,
}

impl<S: G> IvfPq<S> {
    pub fn create(
        path: &Path,
        options: IndexOptions,
        sealed: Vec<Arc<SealedSegment<S>>>,
        growing: Vec<Arc<GrowingSegment<S>>>,
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let ram = make(path, sealed, growing, options);
        let mmap = save(ram, path);
        crate::utils::dir_ops::sync_dir(path);
        Self { mmap }
    }

    pub fn open(path: &Path, options: IndexOptions) -> Self {
        let mmap = open(path, options);
        Self { mmap }
    }

    pub fn len(&self) -> u32 {
        self.mmap.raw.len()
    }

    pub fn vector(&self, i: u32) -> Borrowed<'_, S> {
        self.mmap.raw.vector(i)
    }

    pub fn payload(&self, i: u32) -> Payload {
        self.mmap.raw.payload(i)
    }

    pub fn basic(
        &self,
        vector: Borrowed<'_, S>,
        opts: &SearchOptions,
        filter: impl Filter,
    ) -> BinaryHeap<Reverse<Element>> {
        basic(&self.mmap, vector, opts.ivf_nprobe, filter)
    }

    pub fn vbase<'a>(
        &'a self,
        vector: Borrowed<'a, S>,
        opts: &'a SearchOptions,
        filter: impl Filter + 'a,
    ) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
        vbase(&self.mmap, vector, opts.ivf_nprobe, filter)
    }
}

unsafe impl<S: G> Send for IvfPq<S> {}
unsafe impl<S: G> Sync for IvfPq<S> {}

pub struct IvfRam<S: G> {
    raw: Arc<Raw<S>>,
    quantization: ProductIvfQuantization<S>,
    // ----------------------
    dims: u16,
    // ----------------------
    nlist: u32,
    // ----------------------
    centroids: Vec2<Scalar<S>>,
    ptr: Vec<usize>,
    payloads: Vec<Payload>,
}

unsafe impl<S: G> Send for IvfRam<S> {}
unsafe impl<S: G> Sync for IvfRam<S> {}

pub struct IvfMmap<S: G> {
    raw: Arc<Raw<S>>,
    quantization: ProductIvfQuantization<S>,
    // ----------------------
    dims: u16,
    // ----------------------
    nlist: u32,
    // ----------------------
    centroids: MmapArray<Scalar<S>>,
    ptr: MmapArray<usize>,
    payloads: MmapArray<Payload>,
}

unsafe impl<S: G> Send for IvfMmap<S> {}
unsafe impl<S: G> Sync for IvfMmap<S> {}

impl<S: G> IvfMmap<S> {
    fn centroids(&self, i: u32) -> &[Scalar<S>] {
        let s = i as usize * self.dims as usize;
        let e = (i + 1) as usize * self.dims as usize;
        &self.centroids[s..e]
    }
}

pub fn make<S: G>(
    path: &Path,
    sealed: Vec<Arc<SealedSegment<S>>>,
    growing: Vec<Arc<GrowingSegment<S>>>,
    options: IndexOptions,
) -> IvfRam<S> {
    let VectorOptions { dims, .. } = options.vector;
    let IvfIndexingOptions {
        least_iterations,
        iterations,
        nlist,
        nsample,
        quantization: quantization_opts,
    } = options.indexing.clone().unwrap_ivf();
    let raw = Arc::new(Raw::<S>::create(
        &path.join("raw"),
        options.clone(),
        sealed,
        growing,
    ));
    let n = raw.len();
    let m = std::cmp::min(nsample, n);
    let f = sample(&mut thread_rng(), n as usize, m as usize).into_vec();
    let mut samples = Vec2::new(dims, m as usize);
    for i in 0..m {
        samples[i as usize].copy_from_slice(raw.vector(f[i as usize] as u32).to_vec().as_ref());
        S::elkan_k_means_normalize(&mut samples[i as usize]);
    }
    let mut k_means = ElkanKMeans::<S>::new(nlist as usize, samples);
    for _ in 0..least_iterations {
        k_means.iterate();
    }
    for _ in least_iterations..iterations {
        if k_means.iterate() {
            break;
        }
    }
    let centroids = k_means.finish();
    let mut idx = vec![0usize; n as usize];
    idx.par_iter_mut().enumerate().for_each(|(i, x)| {
        let mut vector = raw.vector(i as u32).for_own();
        S::elkan_k_means_normalize2(&mut vector);
        let mut result = (F32::infinity(), 0);
        for i in 0..nlist as usize {
            let dis = S::elkan_k_means_distance2(vector.for_borrow(), &centroids[i]);
            result = std::cmp::min(result, (dis, i));
        }
        *x = result.1;
    });
    let mut invlists_ids = vec![Vec::new(); nlist as usize];
    let mut invlists_payloads = vec![Vec::new(); nlist as usize];
    for i in 0..n {
        invlists_ids[idx[i as usize]].push(i);
        invlists_payloads[idx[i as usize]].push(raw.payload(i));
    }
    let mut ptr = vec![0usize; nlist as usize + 1];
    for i in 0..nlist {
        ptr[i as usize + 1] = ptr[i as usize] + invlists_ids[i as usize].len();
    }
    let ids = Vec::from_iter((0..nlist).flat_map(|i| &invlists_ids[i as usize]).copied());
    let payloads = Vec::from_iter(
        (0..nlist)
            .flat_map(|i| &invlists_payloads[i as usize])
            .copied(),
    );
    let residuals = {
        let mut residuals = Vec2::new(options.vector.dims, n as usize);
        residuals
            .par_chunks_mut(dims as usize)
            .enumerate()
            .for_each(|(i, v)| {
                for j in 0..dims {
                    v[j as usize] = raw.vector(ids[i]).to_vec()[j as usize]
                        - centroids[idx[ids[i] as usize]][j as usize];
                }
            });
        residuals
    };
    let quantization = ProductIvfQuantization::create(
        &path.join("quantization"),
        options.clone(),
        quantization_opts,
        &residuals,
        &centroids,
    );
    IvfRam {
        raw,
        quantization,
        centroids,
        nlist,
        dims,
        ptr,
        payloads,
    }
}

pub fn save<S: G>(ram: IvfRam<S>, path: &Path) -> IvfMmap<S> {
    let centroids = MmapArray::create(
        &path.join("centroids"),
        (0..ram.nlist)
            .flat_map(|i| &ram.centroids[i as usize])
            .copied(),
    );
    let ptr = MmapArray::create(&path.join("ptr"), ram.ptr.iter().copied());
    let payloads = MmapArray::create(&path.join("payload"), ram.payloads.iter().copied());
    IvfMmap {
        raw: ram.raw,
        quantization: ram.quantization,
        dims: ram.dims,
        nlist: ram.nlist,
        centroids,
        ptr,
        payloads,
    }
}

pub fn open<S: G>(path: &Path, options: IndexOptions) -> IvfMmap<S> {
    let raw = Arc::new(Raw::open(&path.join("raw"), options.clone()));
    let quantization = ProductIvfQuantization::open(
        &path.join("quantization"),
        options.clone(),
        options.indexing.clone().unwrap_ivf().quantization,
        &raw,
    );
    let centroids = MmapArray::open(&path.join("centroids"));
    let ptr = MmapArray::open(&path.join("ptr"));
    let payloads = MmapArray::open(&path.join("payload"));
    let IvfIndexingOptions { nlist, .. } = options.indexing.unwrap_ivf();
    IvfMmap {
        raw,
        quantization,
        dims: options.vector.dims,
        nlist,
        centroids,
        ptr,
        payloads,
    }
}

pub fn basic<S: G>(
    mmap: &IvfMmap<S>,
    vector: Borrowed<'_, S>,
    nprobe: u32,
    mut filter: impl Filter,
) -> BinaryHeap<Reverse<Element>> {
    let dense = vector.to_vec();
    let mut lists = IndexHeap::new(nprobe as usize);
    for i in 0..mmap.nlist {
        let centroid = mmap.centroids(i);
        let distance = S::product_quantization_dense_distance(&dense, centroid);
        if lists.check(distance) {
            lists.push((distance, i));
        }
    }
    let runtime_table = mmap.quantization.init_query(vector.to_vec().as_ref());
    let lists = lists.into_sorted_vec();
    let mut result = BinaryHeap::new();
    for i in lists.iter() {
        let key = i.1 as usize;
        let coarse_dis = i.0;
        let start = mmap.ptr[key];
        let end = mmap.ptr[key + 1];
        for j in start..end {
            let payload = mmap.payloads[j];
            if filter.check(payload) {
                let distance = mmap.quantization.distance_with_codes(
                    vector,
                    j as u32,
                    mmap.centroids(key as u32),
                    key,
                    coarse_dis,
                    &runtime_table,
                );
                result.push(Reverse(Element { distance, payload }));
            }
        }
    }
    result
}

pub fn vbase<'a, S: G>(
    mmap: &'a IvfMmap<S>,
    vector: Borrowed<'a, S>,
    nprobe: u32,
    mut filter: impl Filter + 'a,
) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
    let dense = vector.to_vec();
    let mut lists = IndexHeap::new(nprobe as usize);
    for i in 0..mmap.nlist {
        let centroid = mmap.centroids(i);
        let distance = S::product_quantization_dense_distance(&dense, centroid);
        if lists.check(distance) {
            lists.push((distance, i));
        }
    }
    let runtime_table = mmap.quantization.init_query(vector.to_vec().as_ref());
    let lists = lists.into_sorted_vec();
    let mut result = Vec::new();
    for i in lists.iter() {
        let key = i.1 as usize;
        let coarse_dis = i.0;
        let start = mmap.ptr[key];
        let end = mmap.ptr[key + 1];
        for j in start..end {
            let payload = mmap.payloads[j];
            if filter.check(payload) {
                let distance = mmap.quantization.distance_with_codes(
                    vector,
                    j as u32,
                    mmap.centroids(key as u32),
                    key,
                    coarse_dis,
                    &runtime_table,
                );
                result.push(Element { distance, payload });
            }
        }
    }
    (result, Box::new(std::iter::empty()))
}

pub struct IndexHeap {
    binary_heap: BinaryHeap<(F32, u32)>,
    k: usize,
}

impl IndexHeap {
    pub fn new(k: usize) -> Self {
        assert!(k != 0);
        Self {
            binary_heap: BinaryHeap::new(),
            k,
        }
    }
    pub fn check(&self, distance: F32) -> bool {
        self.binary_heap.len() < self.k || distance < self.binary_heap.peek().unwrap().0
    }
    pub fn push(&mut self, element: (F32, u32)) -> Option<(F32, u32)> {
        self.binary_heap.push(element);
        if self.binary_heap.len() == self.k + 1 {
            self.binary_heap.pop()
        } else {
            None
        }
    }
    pub fn into_sorted_vec(self) -> Vec<(F32, u32)> {
        self.binary_heap.into_sorted_vec()
    }
}

pub struct ProductIvfQuantization<S: G> {
    dims: u16,
    ratio: u16,
    centroids: Vec<Scalar<S>>,
    codes: MmapArray<u8>,
    precomputed_table: Vec<F32>,
}

unsafe impl<S: G> Send for ProductIvfQuantization<S> {}
unsafe impl<S: G> Sync for ProductIvfQuantization<S> {}

impl<S: G> ProductIvfQuantization<S> {
    pub fn codes(&self, i: u32) -> &[u8] {
        let width = self.dims.div_ceil(self.ratio);
        let s = i as usize * width as usize;
        let e = (i + 1) as usize * width as usize;
        &self.codes[s..e]
    }
}

impl<S: G> ProductIvfQuantization<S> {
    pub fn create(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        raw: &Vec2<Scalar<S>>,
        coarse_centroids: &Vec2<Scalar<S>>,
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let QuantizationOptions::Product(quantization_options) = quantization_options else {
            unreachable!()
        };
        let dims = options.vector.dims;
        let ratio = quantization_options.ratio as u16;
        let n = raw.len();
        let m = std::cmp::min(n, quantization_options.sample as usize);
        let samples = {
            let f = sample(&mut thread_rng(), n, m).into_vec();
            let mut samples = Vec2::new(options.vector.dims, m);
            for i in 0..m {
                samples[i].copy_from_slice(&raw[f[i]]);
            }
            samples
        };
        let width = dims.div_ceil(ratio);
        // a temp layout (width * 256 * subdims) for par_chunks_mut
        let mut tmp_centroids = vec![Scalar::<S>::zero(); 256 * dims as usize];
        // this par_for parallelizes over sub quantizers
        tmp_centroids
            .par_chunks_mut(256 * ratio as usize)
            .enumerate()
            .for_each(|(i, v)| {
                // i is the index of subquantizer
                let subdims = std::cmp::min(ratio, dims - ratio * i as u16) as usize;
                let mut subsamples = Vec2::new(subdims as u16, m);
                for j in 0..m {
                    let src = &samples[j][i * ratio as usize..][..subdims];
                    subsamples[j].copy_from_slice(src);
                }
                let mut k_means = ElkanKMeans::<S::ProductQuantizationL2>::new(256, subsamples);
                for _ in 0..25 {
                    if k_means.iterate() {
                        break;
                    }
                }
                let centroid = k_means.finish();
                for j in 0usize..=255 {
                    v[j * subdims..][..subdims].copy_from_slice(&centroid[j]);
                }
            });
        // transform back to normal layout (256 * width * subdims)
        let mut centroids = vec![Scalar::<S>::zero(); 256 * dims as usize];
        centroids
            .par_chunks_mut(dims as usize)
            .enumerate()
            .for_each(|(i, v)| {
                for j in 0..width {
                    let subdims = std::cmp::min(ratio, dims - ratio * j) as usize;
                    v[(j * ratio) as usize..][..subdims].copy_from_slice(
                        &tmp_centroids[(j * ratio) as usize * 256..][i * subdims..][..subdims],
                    );
                }
            });
        let mut codes = vec![0u8; n * width as usize];
        codes
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(id, v)| {
                let vector = raw[id].to_vec();
                let width = dims.div_ceil(ratio);
                for i in 0..width {
                    let subdims = std::cmp::min(ratio, dims - ratio * i);
                    let mut minimal = F32::infinity();
                    let mut target = 0u8;
                    let left = &vector[(i * ratio) as usize..][..subdims as usize];
                    for j in 0u8..=255 {
                        let right = &centroids[j as usize * dims as usize..]
                            [(i * ratio) as usize..][..subdims as usize];
                        let dis = S::ProductQuantizationL2::product_quantization_dense_distance(
                            left, right,
                        );
                        if dis < minimal {
                            minimal = dis;
                            target = j;
                        }
                    }
                    v[i as usize] = target;
                }
            });
        std::fs::write(
            path.join("centroids"),
            serde_json::to_string(&centroids).unwrap(),
        )
        .unwrap();
        let codes = MmapArray::create(&path.join("codes"), codes.into_iter());
        let nlist = coarse_centroids.len();
        let width = dims.div_ceil(ratio);
        let mut precomputed_table = Vec::new();
        precomputed_table.resize(nlist * width as usize * 256, F32::zero());
        precomputed_table
            .par_chunks_mut(width as usize * 256)
            .enumerate()
            .for_each(|(i, v)| {
                let x_c = &coarse_centroids[i];
                for j in 0..width {
                    let subdims = std::cmp::min(ratio, dims - ratio * j);
                    let sub_x_c = &x_c[(j * ratio) as usize..][..subdims as usize];
                    for k in 0usize..256 {
                        let sub_x_r = &centroids[k * dims as usize..][(j * ratio) as usize..]
                            [..subdims as usize];
                        v[j as usize * 256 + k] = squared_norm::<S>(subdims, sub_x_r)
                            + F32(2.0) * inner_product::<S>(subdims, sub_x_c, sub_x_r);
                    }
                }
            });
        std::fs::write(
            path.join("table"),
            serde_json::to_string(&precomputed_table).unwrap(),
        )
        .unwrap();
        crate::utils::dir_ops::sync_dir(path);
        Self {
            dims,
            ratio,
            centroids,
            codes,
            precomputed_table,
        }
    }

    pub fn open(
        path: &Path,
        options: IndexOptions,
        quantization_options: QuantizationOptions,
        _: &Arc<Raw<S>>,
    ) -> Self {
        let QuantizationOptions::Product(quantization_options) = quantization_options else {
            unreachable!()
        };
        let centroids =
            serde_json::from_slice(&std::fs::read(path.join("centroids")).unwrap()).unwrap();
        let codes = MmapArray::open(&path.join("codes"));
        let precomputed_table =
            serde_json::from_slice(&std::fs::read(path.join("table")).unwrap()).unwrap();
        Self {
            dims: options.vector.dims,
            ratio: quantization_options.ratio as _,
            centroids,
            codes,
            precomputed_table,
        }
    }

    // compute term2 at query time
    pub fn init_query(&self, query: &[Scalar<S>]) -> Vec<F32> {
        if S::DISTANCE_KIND == DistanceKind::Cos {
            return Vec::new();
        }
        let dims = self.dims;
        let ratio = self.ratio;
        let width = dims.div_ceil(ratio);
        let mut runtime_table = vec![F32::zero(); width as usize * 256];
        for i in 0..256 {
            for j in 0..width {
                let subdims = std::cmp::min(ratio, dims - ratio * j);
                let sub_query = &query[(j * ratio) as usize..][..subdims as usize];
                let centroid = &self.centroids[i * dims as usize..][(j * ratio) as usize..]
                    [..subdims as usize];
                runtime_table[j as usize * 256 + i] =
                    F32(-1.0) * inner_product::<S>(subdims, sub_query, centroid);
            }
        }
        runtime_table
    }

    // add up all terms given codes
    pub fn distance_with_codes(
        &self,
        lhs: Borrowed<'_, S>,
        rhs: u32,
        delta: &[Scalar<S>],
        key: usize,
        coarse_dis: F32,
        runtime_table: &[F32],
    ) -> F32 {
        if S::DISTANCE_KIND == DistanceKind::Cos {
            return self.distance_with_delta(lhs, rhs, delta);
        }
        let mut result = coarse_dis;
        let codes = self.codes(rhs);
        let width = self.dims.div_ceil(self.ratio);
        let precomputed_table = &self.precomputed_table[key * width as usize * 256..];
        if S::DISTANCE_KIND == DistanceKind::L2 {
            for i in 0..width {
                result += precomputed_table[i as usize * 256 + codes[i as usize] as usize]
                    + F32(2.0) * runtime_table[i as usize * 256 + codes[i as usize] as usize];
            }
        } else if S::DISTANCE_KIND == DistanceKind::Dot {
            for i in 0..width {
                result += runtime_table[i as usize * 256 + codes[i as usize] as usize];
            }
        }
        result
    }

    fn distance_with_delta(&self, lhs: Borrowed<'_, S>, rhs: u32, delta: &[Scalar<S>]) -> F32 {
        let dims = self.dims;
        let ratio = self.ratio;
        let rhs = self.codes(rhs);
        S::product_quantization_distance_with_delta(dims, ratio, &self.centroids, lhs, rhs, delta)
    }
}

pub fn squared_norm<S: G>(dims: u16, vec: &[Scalar<S>]) -> F32 {
    let mut result = F32::zero();
    for i in 0..dims as usize {
        result += F32((vec[i] * vec[i]).to_f32());
    }
    result
}

pub fn inner_product<S: G>(dims: u16, lhs: &[Scalar<S>], rhs: &[Scalar<S>]) -> F32 {
    let mut result = F32::zero();
    for i in 0..dims as usize {
        result += F32((lhs[i] * rhs[i]).to_f32());
    }
    result
}
