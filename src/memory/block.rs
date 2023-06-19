use super::bump::Bump;
use super::BlockOptions;
use crate::memory::BlockType;
use cstr::cstr;
use memmap2::MmapMut;
use parking_lot::Mutex;
use std::alloc::{AllocError, Layout};
use std::fs::{File, OpenOptions};
use std::io::Seek;
use std::os::fd::FromRawFd;

pub struct Block {
    options: BlockOptions,
    mmap: MmapMut,
    file: Mutex<File>,
    bump: Bump,
}

impl Block {
    pub fn open(options: BlockOptions, create: bool) -> anyhow::Result<Self> {
        use BlockType::*;
        anyhow::ensure!(options.block_layout.is_vaild());
        let mut file = match options.block_type {
            OnDisk => tempfile::tempfile(),
            InMemory => unsafe {
                let fd = libc::memfd_create(cstr!("file").as_ptr(), 0);
                if fd != -1 {
                    Ok(File::from_raw_fd(fd))
                } else {
                    Err(std::io::Error::last_os_error())
                }
            },
        }?;
        if create {
            file.set_len(options.block_layout.size as u64)?;
        } else {
            let mut persistent_file = std::fs::OpenOptions::new()
                .read(true)
                .open(&options.persistent_path)?;
            std::io::copy(&mut persistent_file, &mut file)?;
        }
        let mut mmap = unsafe { MmapMut::map_mut(&file) }?;
        let bump = unsafe { Bump::new(options.block_layout, mmap.as_mut_ptr(), create) };
        Ok(Self {
            options,
            mmap,
            file: Mutex::new(file),
            bump,
        })
    }

    pub fn persist(&self) -> anyhow::Result<()> {
        self.mmap.flush()?;
        let mut persistent_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.options.persistent_path)?;
        let mut file = self.file.lock();
        file.seek(std::io::SeekFrom::Start(0))?;
        std::io::copy(&mut *file, &mut persistent_file)?;
        persistent_file.sync_all()?;
        Ok(())
    }

    pub fn lookup(&self, offset: usize) -> *mut u8 {
        assert!(offset <= self.mmap.len());
        unsafe { self.mmap.as_ptr().add(offset).cast_mut() }
    }

    pub fn allocate(&self, layout: Layout) -> Result<usize, AllocError> {
        self.bump.allocate(layout)
    }

    pub unsafe fn deallocate(&self, offset: usize, layout: Layout) {
        self.bump.deallocate(offset, layout)
    }
}

unsafe impl Send for Block {}
unsafe impl Sync for Block {}
