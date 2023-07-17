use std::alloc::{AllocError, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Bump {
    size: usize,
    space: *mut Space,
}

impl Bump {
    pub unsafe fn build(size: usize, addr: *mut u8) -> Self {
        assert!(size >= 4096);
        let space = addr.cast::<Space>();
        space.write(Space {
            cursor: AtomicUsize::new(4096),
            objects: AtomicUsize::new(0),
        });
        Self { size, space }
    }
    pub unsafe fn load(size: usize, addr: *mut u8) -> Self {
        assert!(size >= 4096);
        let space = addr.cast::<Space>();
        Self { size, space }
    }
    pub fn allocate(&self, layout: Layout) -> Result<usize, AllocError> {
        if layout.size() == 0 {
            return Ok(0);
        }
        if layout.align() > 128 {
            return Err(AllocError);
        }
        let mut old = unsafe { (*self.space).cursor.load(Ordering::Relaxed) };
        let offset = loop {
            let offset = (old + layout.align() - 1) & !(layout.align() - 1);
            let new = offset + layout.size();
            if new > self.size {
                return Err(AllocError);
            }
            let exchange = unsafe {
                (*self.space).cursor.compare_exchange_weak(
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
            (*self.space).objects.fetch_add(1, Ordering::Relaxed);
        }
        Ok(offset)
    }
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<usize, AllocError> {
        self.allocate(layout)
    }
    pub unsafe fn deallocate(&self, _offset: usize, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        unsafe {
            (*self.space).objects.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

unsafe impl Send for Bump {}
unsafe impl Sync for Bump {}

#[repr(C)]
struct Space {
    cursor: AtomicUsize,
    objects: AtomicUsize,
}
