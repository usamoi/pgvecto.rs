use super::kmeans::kmeans;
use super::vec2::Vec2;
use crate::algorithms::Vectors;
use crate::memory::Address;
use crate::memory::Allocator;
use crate::memory::PBox;
use crate::memory::Persistent;
use crate::memory::Ptr;
use crate::prelude::Distance;
use crate::prelude::Scalar;
use crate::utils::tcalar::Tcalar;
use crate::utils::unsafe_once::UnsafeOnce;
use crossbeam::atomic::AtomicCell;
use rand::seq::index::sample;
use std::collections::BinaryHeap;
use std::sync::Arc;

struct List {
    centroid: PBox<[Scalar]>,
    head: AtomicCell<Option<usize>>,
}

type Vertex = Option<usize>;

pub struct Root {
    lists: PBox<[List]>,
    vertexs: PBox<[UnsafeOnce<Vertex>]>,
    nprobe: usize,
}

static_assertions::assert_impl_all!(Root: Persistent);

pub struct IvfImpl {
    pub address: Address,
    root: &'static Root,
    vectors: Arc<Vectors>,
    distance: Distance,
}

impl IvfImpl {
    pub fn new(
        vectors: Arc<Vectors>,
        dims: u16,
        distance: Distance,
        n: usize,
        nlist: usize,
        nsample: usize,
        nprobe: usize,
        capacity: usize,
        allocator: Allocator,
    ) -> anyhow::Result<Self> {
        let mut rand = rand::thread_rng();
        let nsamples = std::cmp::min(nsample, n);
        let mut samples = Vec2::new(dims, n);
        for (i, j) in sample(&mut rand, n, nsamples).into_iter().enumerate() {
            samples[i].copy_from_slice(vectors.get_vector(j));
        }
        let centroids = kmeans(distance, dims, &samples, nlist);
        let ptr = PBox::new(
            Root {
                lists: {
                    let mut lists = PBox::new_zeroed_slice(nlist, allocator)?;
                    for i in 0..nlist {
                        lists[i].write(List {
                            centroid: {
                                let mut centroid = unsafe {
                                    PBox::new_zeroed_slice(dims as _, allocator)?.assume_init()
                                };
                                centroid.copy_from_slice(&centroids[i]);
                                centroid
                            },
                            head: AtomicCell::new(None),
                        });
                    }
                    unsafe { lists.assume_init() }
                },
                vertexs: {
                    let vertexs = PBox::new_zeroed_slice(capacity, allocator)?;
                    unsafe { vertexs.assume_init() }
                },
                nprobe,
            },
            allocator,
        )?
        .into_raw();
        Ok(Self {
            address: ptr.address(),
            root: unsafe { ptr.as_ref() },
            vectors,
            distance,
        })
    }
    pub fn load(
        vectors: Arc<Vectors>,
        distance: Distance,
        address: Address,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            address,
            root: unsafe { Ptr::new(address, ()).as_ref() },
            vectors,
            distance,
        })
    }
    pub fn search(
        &self,
        (vector, k): (Box<[Scalar]>, usize),
    ) -> anyhow::Result<Vec<(Scalar, u64)>> {
        let nprobe = self.root.nprobe;
        let mut lists = BinaryHeap::new();
        for (i, list) in self.root.lists.iter().enumerate() {
            let dis = Tcalar(self.distance.distance(&vector, &list.centroid));
            if lists.is_empty() || (dis, i) < lists.peek().copied().unwrap() {
                lists.push((dis, i));
                if lists.len() > nprobe {
                    lists.pop();
                }
            }
        }
        let mut result = BinaryHeap::new();
        for (_, i) in lists.into_iter() {
            let mut cursor = self.root.lists[i].head.load();
            while let Some(node) = cursor {
                let vector = self.vectors.get_vector(node);
                let data = self.vectors.get_data(node);
                let dis = Tcalar(self.distance.distance(&vector, vector));
                if result.is_empty() || (dis, data) < result.peek().copied().unwrap() {
                    result.push((dis, data));
                    if result.len() > k {
                        result.pop();
                    }
                }
                cursor = *self.root.vertexs[node];
            }
        }
        Ok(result
            .into_iter_sorted()
            .map(|(Tcalar(dis), i)| (dis, i))
            .collect())
    }
    pub fn insert(&self, insert: usize) -> anyhow::Result<()> {
        self._insert(insert)?;
        Ok(())
    }
    pub fn _insert(&self, insert: usize) -> anyhow::Result<()> {
        let vertexs = self.root.vertexs.as_ref();
        let vector = self.vectors.get_vector(insert);
        let mut result = (Tcalar(Scalar::INFINITY), 0);
        for (i, list) in self.root.lists.iter().enumerate() {
            let dis = Tcalar(self.distance.distance(&vector, &list.centroid));
            result = std::cmp::min(result, (dis, i));
        }
        loop {
            let next = self.root.lists[result.1].head.load();
            vertexs[insert].set(next);
            if let Ok(_) = self.root.lists[result.1]
                .head
                .compare_exchange(next, Some(insert))
            {
                break;
            }
        }
        Ok(())
    }
}
