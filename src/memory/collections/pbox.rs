use crate::memory::{using, Ptr};
use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::Unsize;
use std::ops::{Deref, DerefMut};

pub struct PBox<T: ?Sized>(Ptr<T>);

impl<T: Sized> PBox<T> {
    pub fn new(t: T, block: usize) -> anyhow::Result<Self> {
        let ptr = using()
            .allocate(block, std::alloc::Layout::new::<T>())?
            .cast::<T>();
        unsafe {
            ptr.as_mut_ptr().write(t);
        }
        Ok(Self(ptr))
    }
}

impl<T: ?Sized> PBox<T> {
    #[allow(dead_code)]
    pub fn coerice<U>(self) -> PBox<U>
    where
        T: Unsize<U>,
        U: ?Sized,
    {
        let raw = Self::into_raw(self);
        let raw = raw.coerice();
        PBox::from_raw(raw)
    }
    pub fn into_raw(self) -> Ptr<T> {
        let raw = self.0;
        std::mem::forget(self);
        raw
    }
    pub fn from_raw(raw: Ptr<T>) -> Self {
        Self(raw)
    }
}

impl<T: ?Sized> Drop for PBox<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::for_value(self.0.as_ref());
            std::ptr::drop_in_place(self.0.as_mut_ptr());
            using().deallocate(self.0.cast(), layout);
        }
    }
}

impl<T> Deref for PBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for PBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut() }
    }
}

impl<T: Debug> Debug for PBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}
