mod block;
mod bump;
mod collections;

pub use collections::*;

use self::block::Block;
use arrayvec::ArrayVec;
use std::alloc::{AllocError, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::marker::Unsize;
use std::ptr::{NonNull, Pointee};

pub const MAX_BLOCKS: usize = 16;

pub unsafe auto trait Persistent {}

impl<T> !Persistent for *const T {}
impl<T> !Persistent for *mut T {}
impl<T> !Persistent for &'_ T {}
impl<T> !Persistent for &'_ mut T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhantomNonpersistent;

impl !Persistent for PhantomNonpersistent {}

#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Address(usize);

impl Address {
    pub fn block(self) -> usize {
        (self.0 >> 57) & ((1usize << 4) - 1)
    }
    pub fn offset(self) -> usize {
        (self.0 >> 0) & ((1usize << 57) - 1)
    }
    pub fn new(block: usize, offset: usize) -> Self {
        assert!(block < (1 << 4));
        assert!(offset < (1 << 57));
        Self(block << 57 | offset << 0)
    }
    #[allow(dead_code)]
    pub fn add(self, bytes: usize) -> Self {
        Self::new(self.block(), self.offset() + bytes)
    }
}

#[repr(C)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ptr<T: ?Sized> {
    address: Address,
    metadata: <T as Pointee>::Metadata,
    _data: PhantomData<T>,
    _nonpersistent: PhantomNonpersistent,
}

impl<T: ?Sized> Copy for Ptr<T> {}

impl<T: ?Sized> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

static_assertions::const_assert_eq!(std::mem::size_of::<Ptr<u8>>(), 8);
static_assertions::const_assert_eq!(std::mem::size_of::<Ptr<[u8]>>(), 16);

unsafe impl<T: ?Sized> Persistent for Ptr<T>
where
    T: Persistent,
    <T as Pointee>::Metadata: Persistent,
{
}

impl<T: ?Sized> Ptr<T> {
    pub fn cast<U>(self) -> Ptr<U>
    where
        U: Sized,
    {
        Ptr {
            address: self.address,
            metadata: (),
            _data: PhantomData,
            _nonpersistent: PhantomNonpersistent,
        }
    }
    #[allow(dead_code)]
    pub fn coerice<U>(self) -> Ptr<U>
    where
        T: Unsize<U>,
        U: ?Sized,
    {
        let metadata = std::ptr::metadata(self.as_mut_ptr() as *mut U);
        Ptr {
            address: self.address,
            metadata,
            _data: PhantomData,
            _nonpersistent: PhantomNonpersistent,
        }
    }
    pub fn from_raw_parts(data_address: Ptr<()>, metadata: <T as Pointee>::Metadata) -> Self {
        Self {
            address: data_address.address,
            metadata,
            _data: PhantomData,
            _nonpersistent: PhantomNonpersistent,
        }
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
    pub fn block(self) -> usize {
        self.address.block()
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
        Self {
            address,
            metadata,
            _data: PhantomData,
            _nonpersistent: PhantomNonpersistent,
        }
    }
    #[allow(dead_code)]
    pub fn add(self, count: usize) -> Self
    where
        T: Sized,
    {
        Self {
            address: self.address.add(count * std::mem::size_of::<T>()),
            metadata: (),
            _data: PhantomData,
            _nonpersistent: PhantomNonpersistent,
        }
    }
}

#[thread_local]
static CONTEXT: Cell<Option<ContextPtr>> = Cell::new(None);

pub unsafe fn given(new_context: ContextPtr) -> impl Drop {
    pub struct Given(());
    impl Drop for Given {
        fn drop(&mut self) {
            let context = CONTEXT.get();
            if context.is_none() {
                unreachable!();
            }
            CONTEXT.set(None);
        }
    }
    let context = CONTEXT.get();
    if context.is_some() {
        panic!("Already given a context.");
    }
    CONTEXT.set(Some(new_context));
    Given(())
}

pub fn using<'a>() -> &'a Context {
    unsafe {
        CONTEXT
            .get()
            .expect("Never given a context to use.")
            .0
            .as_ref()
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct ContextPtr(NonNull<Context>);

unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

impl ContextPtr {
    pub fn new(raw: NonNull<Context>) -> Self {
        Self(raw)
    }
}

pub struct Context {
    blocks: arrayvec::ArrayVec<Block, MAX_BLOCKS>,
}

impl Context {
    pub fn new(options: ContextOptions, create: bool) -> anyhow::Result<Self> {
        let mut blocks = arrayvec::ArrayVec::<Block, MAX_BLOCKS>::new();
        for block_options in options.blocks.iter() {
            blocks.push(Block::open(block_options.clone(), create)?);
        }
        Ok(Self { blocks })
    }
    pub fn persist(&self) -> anyhow::Result<()> {
        for blocks in self.blocks.iter() {
            blocks.persist()?;
        }
        Ok(())
    }
    pub fn lookup<T: ?Sized>(&self, ptr: Ptr<T>) -> *mut T {
        let block = ptr.block();
        let offset = ptr.offset();
        let metadata = ptr.metadata();
        std::ptr::from_raw_parts_mut(self.blocks[block].lookup(offset).cast(), metadata)
    }
    pub fn allocate(&self, block: usize, layout: Layout) -> Result<Ptr<u8>, AllocError> {
        let offset = self.blocks[block].allocate(layout)?;
        let address = Address::new(block, offset);
        let ptr = Ptr::new(address, ());
        Ok(ptr)
    }
    pub unsafe fn deallocate(&self, ptr: Ptr<u8>, layout: Layout) {
        let block = ptr.block();
        let offset = ptr.offset();
        self.blocks[block].deallocate(offset, layout);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextOptions {
    pub blocks: ArrayVec<BlockOptions, MAX_BLOCKS>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockOptions {
    pub persistent_path: String,
    pub block_type: BlockType,
    pub block_layout: BlockLayout,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BlockType {
    OnDisk,
    InMemory,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockLayout {
    pub size: usize,
    pub align: usize,
}

impl BlockLayout {
    pub fn is_vaild(self) -> bool {
        self.size >= 4096
            && self.size % 4096 == 0
            && self.size <= ((1 << 57) - 4096)
            && self.align.is_power_of_two()
            && self.align <= 4096
    }
}
