use super::bump::Bump;
use crate::prelude::Backend;
use cstr::cstr;
use memmap2::MmapMut;
use std::alloc::{AllocError, Layout};
use std::fs::{File, OpenOptions};
use std::os::fd::FromRawFd;

pub struct Block {
    #[allow(dead_code)]
    size: usize,
    path: String,
    mmap: MmapMut,
    bump: Bump,
}

impl Block {
    pub fn build(size: usize, path: String, backend: Backend) -> anyhow::Result<Self> {
        anyhow::ensure!(size % 4096 == 0);
        let file = tempfile(backend)?;
        file.set_len(size as u64)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file) }?;
        mmap.advise(memmap2::Advice::WillNeed)?;
        let bump = unsafe { Bump::build(size, mmap.as_mut_ptr()) };
        Ok(Self {
            size,
            path,
            mmap,
            bump,
        })
    }

    pub fn load(size: usize, path: String, backend: Backend) -> anyhow::Result<Self> {
        anyhow::ensure!(size % 4096 == 0);
        let mut file = tempfile(backend)?;
        let mut persistent_file = std::fs::OpenOptions::new().read(true).open(&path)?;
        std::io::copy(&mut persistent_file, &mut file)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file) }?;
        let bump = unsafe { Bump::load(size, mmap.as_mut_ptr()) };
        Ok(Self {
            size,
            path,
            mmap,
            bump,
        })
    }

    pub fn persist(&self) -> anyhow::Result<()> {
        use std::io::Write;
        let mut persistent_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        persistent_file.write_all(self.mmap.as_ref())?;
        persistent_file.sync_all()?;
        Ok(())
    }

    pub fn address(&self) -> usize {
        self.mmap.as_ptr() as usize
    }

    pub fn allocate(&self, layout: Layout) -> Result<usize, AllocError> {
        self.bump.allocate(layout)
    }

    pub fn allocate_zeroed(&self, layout: Layout) -> Result<usize, AllocError> {
        self.bump.allocate_zeroed(layout)
    }

    pub unsafe fn deallocate(&self, offset: usize, layout: Layout) {
        self.bump.deallocate(offset, layout)
    }
}

fn tempfile(backend: Backend) -> anyhow::Result<File> {
    use Backend::*;
    let file = match backend {
        Disk => tempfile::tempfile()?,
        Ram => unsafe {
            let fd = libc::memfd_create(cstr!("file").as_ptr(), 0);
            if fd != -1 {
                File::from_raw_fd(fd)
            } else {
                anyhow::bail!(std::io::Error::last_os_error());
            }
        },
    };
    Ok(file)
}
