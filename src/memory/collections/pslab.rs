use crate::memory::using;
use crate::memory::Ptr;
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Deref, Index};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct PSlab<T> {
    ptr: Ptr<[UnsafeCell<MaybeUninit<T>>]>,
    len: AtomicUsize,
    inflight: AtomicUsize,
}

impl<T> PSlab<T> {
    pub fn new(capacity: usize, block: usize) -> anyhow::Result<Self> {
        Ok(Self {
            len: AtomicUsize::new(0),
            inflight: AtomicUsize::new(0),
            ptr: {
                let layout = std::alloc::Layout::array::<T>(capacity)?;
                let ptr = using().allocate(block, layout)?.cast::<T>();
                Ptr::<[UnsafeCell<MaybeUninit<T>>]>::from_raw_parts(ptr.cast(), capacity)
            },
        })
    }
    pub fn data(&self) -> &[T] {
        let n = self.len.load(Ordering::Acquire);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, n) }
    }
    pub fn push(&self, data: T) -> anyhow::Result<usize> {
        // If the index is approaching to `usize::MAX`, it will break. But it will not likely happen.
        let index = self.inflight.fetch_add(1, Ordering::AcqRel);
        if index >= self.ptr.metadata() {
            self.inflight.store(self.ptr.metadata(), Ordering::Release);
            anyhow::bail!("The vec is full.");
        }
        unsafe {
            (*self.ptr.as_mut()[index].get()).write(data);
        }
        while let Err(_) =
            self.len
                .compare_exchange_weak(index, index + 1, Ordering::AcqRel, Ordering::Relaxed)
        {
            std::hint::spin_loop();
        }
        Ok(index)
    }
}

unsafe impl<T: Send> Send for PSlab<T> {}
unsafe impl<T: Sync> Sync for PSlab<T> {}

impl<T> Index<usize> for PSlab<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data()[index]
    }
}

impl<T> Deref for PSlab<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data()
    }
}

impl<T> Drop for PSlab<T> {
    fn drop(&mut self) {
        let n = *self.len.get_mut();
        unsafe {
            let to_drop = Ptr::<[T]>::new(self.ptr.address(), n);
            std::ptr::drop_in_place(to_drop.as_mut());
            let to_deallocate = self.ptr;
            let layout = Layout::for_value(self.ptr.as_ref());
            using().deallocate(to_deallocate.cast(), layout);
        }
    }
}

impl<T: Debug> Debug for PSlab<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        list.entries(self.deref());
        list.finish()
    }
}
