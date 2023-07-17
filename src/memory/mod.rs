mod block;
mod bump;
mod parray;
mod pbox;

pub use parray::PArray;
pub use pbox::PBox;

use self::block::Block;
use crate::prelude::Backend;
use serde::{Deserialize, Serialize};
use std::alloc::{AllocError, Layout};
use std::cell::Cell;
use std::fmt::Debug;
use std::marker::Unsize;
use std::ptr::{NonNull, Pointee};
use std::sync::Arc;
use std::thread::{Scope, ScopedJoinHandle};

pub unsafe auto trait Persistent {}

impl<T> !Persistent for *const T {}
impl<T> !Persistent for *mut T {}
impl<T> !Persistent for &'_ T {}
impl<T> !Persistent for &'_ mut T {}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Address(usize);

impl Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:#x})", self.backend(), self.offset())
    }
}

impl Address {
    pub fn backend(self) -> Backend {
        use Backend::*;
        if Ram as usize == (self.0 >> 63) {
            Backend::Ram
        } else {
            Backend::Disk
        }
    }
    pub fn offset(self) -> usize {
        self.0 & ((1usize << 63) - 1)
    }
    pub fn new(backend: Backend, offset: usize) -> Self {
        debug_assert!(offset < (1 << 63));
        Self((backend as usize) << 63 | offset << 0)
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ptr<T: ?Sized> {
    address: Address,
    metadata: <T as Pointee>::Metadata,
}

impl<T: ?Sized> Debug for Ptr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if std::mem::size_of::<<T as Pointee>::Metadata>() == 0 {
            write!(f, "({:?})", self.address())
        } else if std::mem::size_of::<<T as Pointee>::Metadata>() == std::mem::size_of::<usize>() {
            let metadata = unsafe { std::mem::transmute_copy::<_, usize>(&self.metadata()) };
            write!(f, "({:?}, {:#x})", self.address(), metadata)
        } else {
            write!(f, "({:?}, ?)", self.address())
        }
    }
}

impl<T: ?Sized> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Ptr<T> {}

impl<T: ?Sized> Ptr<T> {
    pub fn backend(self) -> Backend {
        self.address.backend()
    }
    pub fn offset(self) -> usize {
        self.address.offset()
    }
    pub fn address(self) -> Address {
        self.address
    }
    pub fn metadata(self) -> <T as Pointee>::Metadata {
        self.metadata
    }
    pub fn new(address: Address, metadata: <T as Pointee>::Metadata) -> Self {
        Self { address, metadata }
    }
    pub fn cast<U: Sized>(self) -> Ptr<U> {
        Ptr::new(self.address, ())
    }
    #[allow(dead_code)]
    pub fn coerice<U: ?Sized>(self) -> Ptr<U>
    where
        T: Unsize<U>,
    {
        let data_address = self.cast();
        let metadata = std::ptr::metadata(self.as_mut_ptr() as *mut U);
        Ptr::from_raw_parts(data_address, metadata)
    }
    pub fn from_raw_parts(data_address: Ptr<()>, metadata: <T as Pointee>::Metadata) -> Self {
        Ptr::new(data_address.address(), metadata)
    }
    pub fn as_ptr(self) -> *const T {
        using().lookup(self)
    }
    pub fn as_mut_ptr(self) -> *mut T {
        using().lookup(self)
    }
    pub unsafe fn as_ref<'a>(self) -> &'a T {
        &*self.as_ptr()
    }
    pub unsafe fn as_mut<'a>(self) -> &'a mut T {
        &mut *self.as_mut_ptr()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Allocator {
    backend: Backend,
}

impl Allocator {
    pub fn allocate(&self, layout: Layout) -> Result<Ptr<()>, AllocError> {
        use Backend::*;
        let offset = match self.backend {
            Ram => using().block_ram.allocate(layout),
            Disk => using().block_disk.allocate(layout),
        }?;
        let address = Address::new(self.backend, offset);
        let ptr = Ptr::new(address, ());
        Ok(ptr)
    }
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<Ptr<()>, AllocError> {
        use Backend::*;
        let offset = match self.backend {
            Ram => using().block_ram.allocate_zeroed(layout),
            Disk => using().block_disk.allocate_zeroed(layout),
        }?;
        let address = Address::new(self.backend, offset);
        let ptr = Ptr::new(address, ());
        Ok(ptr)
    }
}

#[thread_local]
static CONTEXT: Cell<Option<NonNull<Context>>> = Cell::new(None);

pub unsafe fn given(p: NonNull<Context>) -> impl Drop {
    pub struct Given {
        prev: Option<NonNull<Context>>,
    }
    impl Drop for Given {
        fn drop(&mut self) {
            CONTEXT.replace(self.prev.take());
        }
    }
    let given = Given {
        prev: CONTEXT.get(),
    };
    if let Some(prev) = given.prev.clone() {
        assert!(p == prev);
    }
    CONTEXT.replace(Some(p));
    given
}

pub fn using<'a>() -> &'a Context {
    unsafe {
        CONTEXT
            .get()
            .expect("Never given a context to use.")
            .as_ref()
    }
}

pub struct Context {
    block_ram: Block,
    block_disk: Block,
    offsets: [usize; 2],
}

impl Context {
    pub fn build(options: ContextOptions) -> anyhow::Result<Arc<Self>> {
        let block_ram = Block::build(options.block_ram.0, options.block_ram.1, Backend::Ram)?;
        let block_disk = Block::build(options.block_disk.0, options.block_disk.1, Backend::Disk)?;
        Ok(Arc::new(Self {
            offsets: [block_ram.address(), block_disk.address()],
            block_ram,
            block_disk,
        }))
    }
    pub fn load(options: ContextOptions) -> anyhow::Result<Arc<Self>> {
        let block_ram = Block::load(options.block_ram.0, options.block_ram.1, Backend::Ram)?;
        let block_disk = Block::load(options.block_disk.0, options.block_disk.1, Backend::Disk)?;
        Ok(Arc::new(Self {
            offsets: [block_ram.address(), block_disk.address()],
            block_ram,
            block_disk,
        }))
    }
    pub fn allocator(&self, backend: Backend) -> Allocator {
        Allocator { backend }
    }
    pub fn persist(&self) -> anyhow::Result<()> {
        self.block_ram.persist()?;
        self.block_disk.persist()?;
        Ok(())
    }
    pub fn lookup<T: ?Sized>(&self, ptr: Ptr<T>) -> *mut T {
        let metadata = ptr.metadata();
        let data_address = (self.offsets[ptr.backend() as usize] + ptr.offset()) as _;
        std::ptr::from_raw_parts_mut(data_address, metadata)
    }
    pub unsafe fn deallocate(&self, ptr: Ptr<()>, layout: Layout) {
        use Backend::*;
        let offset = ptr.offset();
        match ptr.backend() {
            Ram => self.block_ram.deallocate(offset, layout),
            Disk => self.block_disk.deallocate(offset, layout),
        }
    }
    pub fn scope<'env, F, T>(&self, f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope ContextScope<'scope, 'env>) -> T,
    {
        std::thread::scope(|scope| {
            f(unsafe { std::mem::transmute::<&Scope, &ContextScope>(scope) })
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextOptions {
    pub block_ram: (usize, String),
    pub block_disk: (usize, String),
}

#[repr(transparent)]
pub struct ContextScope<'scope, 'env: 'scope>(Scope<'scope, 'env>);

impl<'scope, 'env: 'scope> ContextScope<'scope, 'env> {
    pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        struct AssertSend<T>(T);
        impl<T> AssertSend<T> {
            fn cosume(self) -> T {
                self.0
            }
        }
        unsafe impl<T> Send for AssertSend<T> {}
        let wrapped = AssertSend(CONTEXT.get().unwrap());
        self.0.spawn(move || {
            let context = wrapped.cosume();
            let _given = unsafe { given(context) };
            f()
        })
    }
}
