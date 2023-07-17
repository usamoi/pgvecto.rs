use crate::memory::using;
use crate::memory::Address;
use crate::memory::PBox;
use crate::memory::Persistent;
use crate::memory::Ptr;
use crate::prelude::BincodeDeserialize;
use crate::prelude::BincodeSerialize;
use crate::prelude::Options;
use crate::prelude::Scalar;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

type Boxed<T> = PBox<[UnsafeCell<MaybeUninit<T>>]>;

pub struct Root {
    len: AtomicUsize,
    inflight: AtomicUsize,
    // ----------------------
    data: Boxed<u64>,
    vector: Boxed<Scalar>,
}

static_assertions::assert_impl_all!(Root: Persistent);

pub struct Vectors {
    root: &'static Root,
    dims: u16,
    capacity: usize,
}

impl Vectors {
    pub fn build(options: Options) -> anyhow::Result<(Self, Vec<u8>)> {
        let allocator = using().allocator(options.backend_vectors);
        let ptr = PBox::new(
            unsafe {
                let len_data = options.capacity;
                let len_vector = options.capacity * options.dims as usize;
                Root {
                    len: AtomicUsize::new(0),
                    inflight: AtomicUsize::new(0),
                    data: PBox::new_uninit_slice(len_data, allocator)?.assume_init(),
                    vector: PBox::new_uninit_slice(len_vector, allocator)?.assume_init(),
                }
            },
            allocator,
        )?
        .into_raw();
        let root = unsafe { ptr.as_ref() };
        let address = ptr.address();
        let persistent = VectorsPersistent { address };
        Ok((
            Self {
                root,
                dims: options.dims,
                capacity: options.capacity,
            },
            persistent.serialize()?,
        ))
    }
    pub fn load(options: Options, persistent: Vec<u8>) -> anyhow::Result<Self> {
        let VectorsPersistent { address } = persistent.deserialize()?;
        Ok(Self {
            root: unsafe { Ptr::new(address, ()).as_ref() },
            capacity: options.capacity,
            dims: options.dims,
        })
    }
    pub fn put(&self, data: u64, vector: &[Scalar]) -> anyhow::Result<usize> {
        // If the index is approaching to `usize::MAX`, it will break. But it will not likely happen.
        let i = self.root.inflight.fetch_add(1, Ordering::AcqRel);
        if i >= self.capacity {
            self.root.inflight.store(self.capacity, Ordering::Release);
            anyhow::bail!("Full.");
        }
        unsafe {
            let uninit_data = &mut *self.root.data[i].get();
            MaybeUninit::write(uninit_data, data);
            let uninit_vector =
                assume_mutable(&self.root.vector[i * self.dims as usize..][..self.dims as usize]);
            MaybeUninit::write_slice(uninit_vector, vector);
        }
        while let Err(_) =
            self.root
                .len
                .compare_exchange_weak(i, i + 1, Ordering::AcqRel, Ordering::Relaxed)
        {
            std::hint::spin_loop();
        }
        Ok(i)
    }
    pub fn len(&self) -> usize {
        self.root.len.load(Ordering::Acquire)
    }
    pub fn get_data(&self, i: usize) -> u64 {
        unsafe { (*self.root.data[i].get()).assume_init_read() }
    }
    pub fn get_vector(&self, i: usize) -> &[Scalar] {
        unsafe {
            assume_immutable_init(&self.root.vector[i * self.dims as usize..][..self.dims as usize])
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct VectorsPersistent {
    address: Address,
}

unsafe fn assume_mutable<T>(slice: &[UnsafeCell<T>]) -> &mut [T] {
    let p = slice.as_ptr().cast::<UnsafeCell<T>>() as *mut T;
    std::slice::from_raw_parts_mut(p, slice.len())
}

unsafe fn assume_immutable_init<T>(slice: &[UnsafeCell<MaybeUninit<T>>]) -> &[T] {
    let p = slice.as_ptr().cast::<UnsafeCell<T>>() as *const T;
    std::slice::from_raw_parts(p, slice.len())
}
