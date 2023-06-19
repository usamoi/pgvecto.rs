use crate::memory::{using, Ptr};
use std::marker::Unsize;
use std::ops::Deref;

#[derive(Clone)]
pub struct PStatic<T: ?Sized>(Ptr<T>);

impl<T: Sized> PStatic<T> {
    #[allow(dead_code)]
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

impl<T: ?Sized> PStatic<T> {
    #[allow(dead_code)]
    pub fn coerice<U>(self) -> PStatic<U>
    where
        T: Unsize<U>,
        U: ?Sized,
    {
        let raw = self.0;
        let raw = raw.coerice();
        PStatic(raw)
    }
}

impl<T> Deref for PStatic<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}
