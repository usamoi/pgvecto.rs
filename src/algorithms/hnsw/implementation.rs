use super::semaphore::Semaphore;
use super::visited::Visited;
use crate::memory::given;
use crate::memory::Address;
use crate::memory::ContextPtr;
use crate::memory::PArray;
use crate::memory::PBox;
use crate::memory::PSlab;
use crate::memory::Persistent;
use crate::memory::Ptr;
use crate::prelude::*;
use parking_lot::RwLock;
use parking_lot::RwLockReadGuard;
use parking_lot::RwLockWriteGuard;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::RangeInclusive;

#[derive(Debug)]
pub struct Root {
    entry: RwLock<Option<usize>>,
    graph: PSlab<Vertex>,
}

static_assertions::assert_impl_all!(Root: Persistent);

pub struct Implementation {
    distance: Distance,
    #[allow(dead_code)]
    dimensions: u16,
    m: usize,
    ef_construction: usize,
    max_level: usize,
    pub root: Ptr<Root>,
    visited: Semaphore<Visited>,
    block_vectors: usize,
    block_index: usize,
    context: ContextPtr,
}

unsafe impl Send for Implementation {}
unsafe impl Sync for Implementation {}

impl Implementation {
    pub fn load(
        distance: Distance,
        dimensions: u16,
        capacity: usize,
        max_threads: usize,
        m: usize,
        ef_construction: usize,
        max_level: usize,
        blocks: Vec<usize>,
        address: Address,
        context: ContextPtr,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            distance,
            dimensions,
            m,
            ef_construction,
            max_level,
            root: Ptr::new(address, ()),
            visited: {
                let semaphore = Semaphore::<Visited>::new();
                for _ in 0..max_threads {
                    semaphore.push(Visited::new(capacity));
                }
                semaphore
            },
            block_vectors: blocks[0],
            block_index: blocks[1],
            context,
        })
    }
    pub fn new(
        dimensions: u16,
        capacity: usize,
        distance: Distance,
        max_threads: usize,
        m: usize,
        ef_construction: usize,
        max_level: usize,
        blocks: Vec<usize>,
        context: ContextPtr,
    ) -> anyhow::Result<Self> {
        let _given = unsafe { given(context) };
        let block_vectors = blocks[0];
        let block_index = blocks[1];
        let root = PBox::new(
            Root {
                entry: RwLock::new(None),
                graph: PSlab::new(capacity, block_index)?,
            },
            block_index,
        )?;
        Ok(Self {
            dimensions,
            distance,
            root: PBox::into_raw(root),
            visited: {
                let semaphore = Semaphore::<Visited>::new();
                for _ in 0..max_threads {
                    semaphore.push(Visited::new(capacity));
                }
                semaphore
            },
            m,
            ef_construction,
            max_level,
            block_vectors,
            block_index,
            context,
        })
    }
    pub fn search(&self, (vector, k): (Vec<Scalar>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        let _given = unsafe { given(self.context) };
        let root = unsafe { self.root.as_ref() };
        let entry = root.entry.read().clone();
        let Some(u) = entry else { return Ok(Vec::new()) };
        let top = root.graph[u].levels();
        let u = self._go(1..=top, u, &vector);
        let mut visited = self.visited.acquire();
        let mut result = self._search(&mut visited, &vector, u, k, 0);
        result.sort();
        Ok(result
            .iter()
            .map(|&(score, u)| (score.0, root.graph[u].data()))
            .collect::<Vec<_>>())
    }
    pub fn insert(&self, insert: (Vec<Scalar>, u64)) -> anyhow::Result<()> {
        let _given = unsafe { given(self.context) };
        let root = unsafe { self.root.as_ref() };
        let mut visited = self.visited.acquire();
        Ok(self._insert(&mut visited, insert)?)
    }
    fn _go(&self, levels: RangeInclusive<u8>, u: usize, target: &[Scalar]) -> usize {
        let root = unsafe { self.root.as_ref() };
        let mut u = u;
        let mut u_dis = self._dist0(u, target);
        for i in levels.rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let guard = root.graph[u].read(i);
                for (_, v) in guard.iter().copied() {
                    let v_dis = self._dist0(v, target);
                    if v_dis < u_dis {
                        u = v;
                        u_dis = v_dis;
                        changed = true;
                    }
                }
            }
        }
        u
    }
    fn _insert(
        &self,
        visited: &mut Visited,
        (vector, data): (Vec<Scalar>, u64),
    ) -> anyhow::Result<()> {
        let root = unsafe { self.root.as_ref() };
        let levels = generate_random_levels(self.m, self.max_level);
        let entry;
        let lock = {
            let root = unsafe { self.root.as_ref() };
            let cond = move |global: Option<_>| {
                if let Some(u) = global {
                    let g = &root.graph;
                    g[u].levels() < levels
                } else {
                    true
                }
            };
            let lock = root.entry.read();
            if cond(*lock) {
                drop(lock);
                let lock = root.entry.write();
                entry = *lock;
                if cond(*lock) {
                    Some(lock)
                } else {
                    None
                }
            } else {
                entry = *lock;
                None
            }
        };
        let Some(mut u) = entry else {
            let mut layers = Vec::new();
            for _ in 0..=levels as usize {
                layers.push(Vec::new());
            }
            let vertex = Vertex::new(data, vector, &layers, self.block_vectors, self.block_index, self.m)?;
            let pointer = root.graph.push(vertex)?;
            *lock.unwrap() = Some(pointer);
            return Ok(());
        };
        let top = root.graph[u].levels();
        if top > levels {
            u = self._go(levels + 1..=top, u, &vector);
        }
        let mut layers = Vec::with_capacity(1 + levels as usize);
        for i in (0..=std::cmp::min(levels, top)).rev() {
            let mut layer = self._search(visited, &vector, u, self.ef_construction, i);
            layer.sort();
            self._select0(&mut layer, size_of_a_layer(self.m, i))?;
            u = layer.first().unwrap().1;
            layers.push(layer);
        }
        layers.reverse();
        layers.resize_with(1 + levels as usize, || Vec::new());
        let backup = layers.iter().map(|x| x.to_vec()).collect::<Vec<_>>();
        let vertex = Vertex::new(
            data,
            vector,
            &layers,
            self.block_vectors,
            self.block_index,
            self.m,
        )?;
        let pointer = root.graph.push(vertex)?;
        for (i, layer) in backup.into_iter().enumerate() {
            let i = i as u8;
            for (n_dis, n) in layer.iter().copied() {
                let mut guard = root.graph[n].write(i);
                orderedly_insert(&mut guard, (n_dis, pointer))?;
                self._select1(&mut guard, size_of_a_layer(self.m, i))?;
            }
        }
        if let Some(mut lock) = lock {
            *lock = Some(pointer);
        }
        Ok(())
    }
    fn _select0(&self, v: &mut Vec<(Tcalar, usize)>, size: usize) -> anyhow::Result<()> {
        assert!(v.is_sorted());
        if v.len() <= size {
            return Ok(());
        }
        let cloned = v.to_vec();
        v.clear();
        for (u_dis, u) in cloned.iter().copied() {
            if v.len() == size {
                break;
            }
            let check = v
                .iter()
                .map(|&(_, v)| self._dist1(u, v))
                .all(|dist| dist > u_dis);
            if check {
                v.push((u_dis, u));
            }
        }
        Ok(())
    }
    fn _select1(&self, v: &mut PArray<(Tcalar, usize)>, size: usize) -> anyhow::Result<()> {
        if v.len() <= size {
            return Ok(());
        }
        let cloned = v.to_vec();
        v.clear();
        for (u_dis, u) in cloned.iter().copied() {
            if v.len() == size {
                break;
            }
            let check = v
                .iter()
                .map(|&(_, v)| self._dist1(u, v))
                .all(|dist| dist > u_dis);
            if check {
                v.push((u_dis, u)).unwrap();
            }
        }
        Ok(())
    }
    fn _search(
        &self,
        visited: &mut Visited,
        target: &[Scalar],
        s: usize,
        k: usize,
        i: u8,
    ) -> Vec<(Tcalar, usize)> {
        let root = unsafe { self.root.as_ref() };
        assert!(k > 0);
        let mut bound = Tcalar(Scalar::INFINITY);
        let mut visited = visited.new_version();
        let mut candidates = BinaryHeap::<Reverse<(Tcalar, usize)>>::new();
        let mut results = BinaryHeap::<(Tcalar, usize)>::new();
        let s_dis = self._dist0(s, target);
        visited.set(s);
        candidates.push(Reverse((s_dis, s)));
        results.push((s_dis, s));
        if results.len() == k + 1 {
            results.pop();
        }
        if results.len() == k {
            bound = results.peek().unwrap().0;
        }
        while let Some(Reverse((u_dis, u))) = candidates.pop() {
            if u_dis > bound {
                break;
            }
            let guard = root.graph[u].read(i);
            for (_, v) in guard.iter().copied() {
                if visited.test(v) {
                    continue;
                }
                visited.set(v);
                let v_dis = self._dist0(v, target);
                if v_dis > bound {
                    continue;
                }
                candidates.push(Reverse((v_dis, v)));
                results.push((v_dis, v));
                if results.len() == k + 1 {
                    results.pop();
                }
                if results.len() == k {
                    bound = results.peek().unwrap().0;
                }
            }
        }
        results.into_vec()
    }
    fn _dist0(&self, u: usize, target: &[Scalar]) -> Tcalar {
        let root = unsafe { self.root.as_ref() };
        let u = root.graph[u].vector();
        Tcalar(self.distance.distance(u, target))
    }
    fn _dist1(&self, u: usize, v: usize) -> Tcalar {
        let root = unsafe { self.root.as_ref() };
        let u = root.graph[u].vector();
        let v = root.graph[v].vector();
        Tcalar(self.distance.distance(u, v))
    }
}

#[derive(Debug)]
struct Vertex {
    data: u64,
    vector: PArray<Scalar>,
    layers: PArray<RwLock<PArray<(Tcalar, usize)>>>,
}

impl Vertex {
    fn new(
        data: u64,
        vector: Vec<Scalar>,
        layers: &[Vec<(Tcalar, usize)>],
        block_vectors: usize,
        block_index: usize,
        m: usize,
    ) -> anyhow::Result<Self> {
        assert!(layers.len() != 0);
        Ok(Self {
            data,
            vector: PArray::from_rust(vector.capacity(), vector, block_vectors)?,
            layers: {
                let mut r = PArray::new(layers.len(), block_index)?;
                for i in 0..layers.len() {
                    r.push(RwLock::new(PArray::from_rust(
                        1 + size_of_a_layer(m, i as u8),
                        layers[i].clone(),
                        block_index,
                    )?))?;
                }
                r
            },
        })
    }
    fn data(&self) -> u64 {
        self.data
    }
    fn vector(&self) -> &[Scalar] {
        &self.vector
    }
    fn levels(&self) -> u8 {
        self.layers.len() as u8 - 1
    }
    fn read(&self, i: u8) -> RwLockReadGuard<'_, PArray<(Tcalar, usize)>> {
        self.layers[i as usize].read()
    }
    fn write(&self, i: u8) -> RwLockWriteGuard<'_, PArray<(Tcalar, usize)>> {
        self.layers[i as usize].write()
    }
}

// "Tcalar" is "Total-ordering scalar".
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
struct Tcalar(Scalar);

impl PartialEq for Tcalar {
    fn eq(&self, other: &Self) -> bool {
        use std::cmp::Ordering;
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Tcalar {}

impl PartialOrd for Tcalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tcalar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn generate_random_levels(m: usize, max_level: usize) -> u8 {
    let factor = 1.0 / (m as f64).ln();
    let mut rng = rand::thread_rng();
    let x = -rng.gen_range(0.0f64..1.0).ln() * factor;
    x.round().min(max_level as f64) as u8
}

fn size_of_a_layer(m: usize, i: u8) -> usize {
    if i == 0 {
        m * 2
    } else {
        m
    }
}

pub fn orderedly_insert<T: Ord>(a: &mut PArray<T>, element: T) -> anyhow::Result<usize> {
    let (Ok(index) | Err(index)) = a.binary_search(&element);
    a.insert(index, element)?;
    Ok(index)
}
