use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
use crc32fast::hash as crc32;
use std::fs::{File, OpenOptions};
use std::io::Error;
use std::io::ErrorKind;
use std::io::Read;
use std::io::Write;
use std::path::Path;

/*
+----------+-----------+---------+
| CRC (4B) | Size (2B) | Payload |
+----------+-----------+---------+
*/

#[derive(Debug, Clone, Copy)]
pub enum WalStatus {
    Read,
    Truncate,
    Write,
    Flush,
}

pub struct Wal {
    file: File,
    offset: usize,
    status: WalStatus,
}

impl Wal {
    pub fn open(path: impl AsRef<Path>) -> Self {
        use WalStatus::*;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(path)
            .unwrap();
        Self {
            file,
            offset: 0,
            status: Read,
        }
    }
    pub fn create(path: impl AsRef<Path>) -> Self {
        use WalStatus::*;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(path)
            .unwrap();
        Self {
            file,
            offset: 0,
            status: Write,
        }
    }
    pub fn read(&mut self) -> Option<Vec<u8>> {
        use WalStatus::*;
        let Read = self.status else { panic!("Operation not permitted.") };
        let maybe_error: Result<Vec<u8>, Error> = try {
            let crc = self.file.read_u32::<LE>()?;
            let len = self.file.read_u16::<LE>()?;
            let mut data = vec![0u8; len as usize];
            self.file.read_exact(&mut data)?;
            if crc32(&data) == crc {
                self.offset += 4 + 2 + data.len();
                data
            } else {
                let e = Error::new(ErrorKind::UnexpectedEof, "Bad crc.");
                Err(e)?
            }
        };
        match maybe_error {
            Ok(data) => Some(data),
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                self.status = WalStatus::Truncate;
                None
            }
            Err(error) => panic!("IO Error: {}", error),
        }
    }
    pub fn truncate(&mut self) {
        use WalStatus::*;
        let Truncate = self.status else { panic!("Operation not permitted.") };
        self.file.set_len(self.offset as _).unwrap();
        self.file.flush().unwrap();
        self.status = WalStatus::Flush;
    }
    pub fn write(&mut self, bytes: &[u8]) {
        use WalStatus::*;
        let (Write | Flush) = self.status else { panic!("Operation not permitted.") };
        self.file.write_u32::<LE>(crc32(bytes)).unwrap();
        self.file.write_u16::<LE>(bytes.len() as _).unwrap();
        self.file.write_all(&bytes).unwrap();
        self.offset += 4 + 2 + bytes.len();
        self.status = WalStatus::Write;
    }
    pub fn flush(&mut self) {
        use WalStatus::*;
        let (Write | Flush) = self.status else { panic!("Operation not permitted.") };
        self.file.sync_all().unwrap();
        self.status = WalStatus::Flush;
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        use WalStatus::*;
        if let Write | Flush = self.status {
            self.file.sync_all().unwrap();
        }
    }
}
