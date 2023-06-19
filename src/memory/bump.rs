use super::BlockLayout;
use std::alloc::{AllocError, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Bump {
    layout: BlockLayout,
    bump0: *mut Bump0,
}

impl Bump {
    pub unsafe fn new(layout: BlockLayout, addr: *mut u8, create: bool) -> Self {
        assert!(layout.size >= 4096 && layout.align >= 8 && layout.align <= 4096);
        let bump0 = addr.cast::<Bump0>();
        if create {
            bump0.write(Bump0 {
                objects: AtomicUsize::new(0),
                cursor: AtomicUsize::new(4096),
            });
        }
        Self { layout, bump0 }
    }
    pub fn allocate(&self, layout: Layout) -> Result<usize, AllocError> {
        if layout.size() == 0 {
            return Ok(0);
        }
        if self.layout.align < layout.align() {
            return Err(AllocError);
        }
        let mut old = unsafe { (*self.bump0).cursor.load(Ordering::Relaxed) };
        let offset = loop {
            let offset = (old + layout.align() - 1) & !(layout.align() - 1);
            let new = offset + layout.size();
            if new > self.layout.size {
                return Err(AllocError);
            }
            let exchange = unsafe {
                (*self.bump0).cursor.compare_exchange_weak(
                    old,
                    new,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
            };
            let Err(_old) = exchange else { break offset };
            old = _old;
        };
        unsafe {
            (*self.bump0).objects.fetch_add(1, Ordering::Relaxed);
        }
        Ok(offset)
    }
    pub unsafe fn deallocate(&self, _offset: usize, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        unsafe {
            (*self.bump0).objects.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[repr(C, align(4096))]
pub struct Bump0 {
    pub objects: AtomicUsize,
    pub cursor: AtomicUsize,
}

static_assertions::assert_eq_size!(Bump0, [u8; 4096]);
