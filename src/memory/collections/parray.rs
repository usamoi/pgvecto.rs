use crate::memory::{using, Ptr};
use std::alloc::Layout;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

pub struct PArray<T> {
    ptr: Ptr<[MaybeUninit<T>]>,
    len: usize,
}

impl<T> PArray<T> {
    pub fn new(capacity: usize, block: usize) -> anyhow::Result<Self> {
        let layout = std::alloc::Layout::array::<T>(capacity)?;
        let ptr = using().allocate(block, layout)?.cast::<T>();
        Ok(Self {
            ptr: Ptr::from_raw_parts(ptr.cast(), capacity),
            len: 0,
        })
    }
    pub fn from_rust(new_capacity: usize, mut this: Vec<T>, block: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(new_capacity >= this.len());
        let mut result = PArray::<T>::new(new_capacity, block)?;
        for element in this.drain(..) {
            result.push(element)?;
        }
        Ok(result)
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.ptr.metadata()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn insert(&mut self, index: usize, element: T) -> anyhow::Result<()> {
        assert!(index <= self.len);
        if self.capacity() == self.len {
            anyhow::bail!("The vector is full.");
        }
        unsafe {
            if index == self.len {
                self.ptr.as_mut()[index].write(element);
                self.len += 1;
            } else {
                let p = self.ptr.as_mut().as_ptr().add(index).cast_mut();
                std::ptr::copy(p, p.add(1), self.len - index);
                self.ptr.as_mut()[index].write(element);
                self.len += 1;
            }
        }
        Ok(())
    }
    pub fn push(&mut self, element: T) -> anyhow::Result<()> {
        if self.capacity() == self.len {
            anyhow::bail!("The vector is full.");
        }
        unsafe {
            self.ptr.as_mut()[self.len].write(element);
            self.len += 1;
        }
        Ok(())
    }
    #[allow(dead_code)]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value;
        unsafe {
            self.len -= 1;
            value = self.ptr.as_mut()[self.len].assume_init_read();
        }
        Some(value)
    }
}

impl<T> Deref for PArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { Ptr::from_raw_parts(self.ptr.cast(), self.len).as_ref() }
    }
}

impl<T> DerefMut for PArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Ptr::from_raw_parts(self.ptr.cast(), self.len).as_mut() }
    }
}

impl<T> Drop for PArray<T> {
    fn drop(&mut self) {
        let n = self.len;
        unsafe {
            let to_drop = Ptr::<[T]>::new(self.ptr.address(), n);
            std::ptr::drop_in_place(to_drop.as_mut());
            let to_deallocate = self.ptr;
            let layout = Layout::for_value(self.ptr.as_ref());
            using().deallocate(to_deallocate.cast(), layout);
        }
    }
}

impl<T: Debug> Debug for PArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        list.entries(self.deref());
        list.finish()
    }
}
