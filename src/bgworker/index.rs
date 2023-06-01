use crate::bgworker::algo_hnsw::AlgoHnsw;
use crate::bgworker::wal::Wal;
use crate::bgworker::Options;
use crate::postgres::{HeapPointer, Id, Scalar};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;

pub struct Index {
    #[allow(dead_code)]
    id: Id,
    algo: AlgoHnsw,
    version: IndexVersion,
    wal: Wal,
}

impl Index {
    pub fn build(
        id: Id,
        options: Options,
        mut parts: impl Iterator<Item = (Vec<Scalar>, HeapPointer)>,
    ) -> Self {
        remove_file_if_exists(format!("{}_wal", id.as_u32())).unwrap();
        remove_file_if_exists(format!("{}_data", id.as_u32())).unwrap();
        let mut algo = AlgoHnsw::build(options.clone());
        while let Some((vector, p)) = parts.next() {
            algo.insert(&vector, p.into_u48() << 16);
        }
        algo.save(format!("{}_data", id.as_u32()));
        let mut wal = Wal::create(format!("{}_wal", id.as_u32()));
        wal.write(&bincode::serialize(&MetaLog { options }).unwrap());
        wal.flush();
        Self {
            id,
            algo,
            version: IndexVersion::new(),
            wal,
        }
    }

    pub fn load(id: Id) -> Self {
        let mut wal = Wal::open(format!("{}_wal", id.as_u32()));
        let MetaLog { options } = bincode::deserialize::<MetaLog>(&wal.read().unwrap()).unwrap();
        let mut algo = AlgoHnsw::load(options, format!("{}_data", id.as_u32()));
        let mut version = IndexVersion::new();
        loop {
            let Some(bytes) = wal.read() else { break };
            let tuple = bincode::deserialize::<ReplayLog>(&bytes).unwrap();
            match tuple {
                ReplayLog::Insert { vector, p } => {
                    algo.insert(&vector, version.insert(p));
                }
                ReplayLog::Delete { p } => {
                    version.remove(p);
                }
            }
        }
        wal.truncate();
        wal.flush();
        Self {
            id,
            algo,
            version,
            wal,
        }
    }

    pub fn insert(&mut self, (vector, p): (Vec<Scalar>, HeapPointer)) {
        self.algo.insert(&vector, self.version.insert(p));
        let bytes = bincode::serialize(&ReplayLog::Insert { vector, p }).unwrap();
        self.wal.write(&bytes);
        self.wal.flush();
    }

    pub fn bulkdelete(&mut self, deletes: &[HeapPointer]) {
        for p in deletes.iter().copied() {
            self.version.remove(p);
            let bytes = bincode::serialize(&ReplayLog::Delete { p }).unwrap();
            self.wal.write(&bytes);
        }
        self.wal.flush();
    }

    pub fn search(&mut self, (vector, k): (Vec<Scalar>, usize)) -> Vec<HeapPointer> {
        let result = self.algo.search(&vector, k);
        result
            .into_iter()
            .filter_map(|x| self.version.filter(x))
            .collect()
    }
}

struct IndexVersion {
    data: HashMap<HeapPointer, (u16, bool)>,
}

impl IndexVersion {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    pub fn filter(&self, x: u64) -> Option<HeapPointer> {
        let p = HeapPointer::from_u48(x >> 16);
        let v = x as u16;
        if let Some((cv, cve)) = self.data.get(&p) {
            debug_assert!(v < *cv || (v == *cv && *cve));
            if v == *cv {
                Some(p)
            } else {
                None
            }
        } else {
            debug_assert!(v == 0);
            Some(p)
        }
    }
    pub fn insert(&mut self, p: HeapPointer) -> u64 {
        if let Some((cv, cve)) = self.data.get_mut(&p) {
            debug_assert!(*cve == false);
            *cve = true;
            p.into_u48() << 16 | *cv as u64
        } else {
            self.data.insert(p, (0, true));
            p.into_u48() << 16 | 0
        }
    }
    pub fn remove(&mut self, p: HeapPointer) {
        if let Some((cv, cve)) = self.data.get_mut(&p) {
            if *cve == true {
                *cv = *cv + 1;
                *cve = false;
            }
        } else {
            self.data.insert(p, (1, false));
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct MetaLog {
    options: Options,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
enum ReplayLog {
    Insert { vector: Vec<Scalar>, p: HeapPointer },
    Delete { p: HeapPointer },
}

fn remove_file_if_exists(path: impl AsRef<Path>) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}
