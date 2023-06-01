use crate::bgworker::{CommonOptions, Options};
use crate::bridge::ffi;
use crate::postgres::Scalar;
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

pub struct AlgoHnsw {
    #[allow(dead_code)]
    options_common: CommonOptions,
    #[allow(dead_code)]
    options_algo: String,
    binding: *mut ffi::Binding,
}

unsafe impl Sync for AlgoHnsw {}
unsafe impl Send for AlgoHnsw {}

impl AlgoHnsw {
    pub fn build(options: Options) -> Self {
        let options_common = options.common;
        let options_algo = options.algo;
        let dims = options_common.dims as usize;
        let binding = unsafe { ffi::binding_new(dims) };
        Self {
            binding,
            options_common,
            options_algo,
        }
    }
    pub fn load(options: Options, path: impl AsRef<Path>) -> Self {
        let options_common = options.common;
        let options_algo = options.algo;
        let dims = options_common.dims as usize;
        let binding;
        unsafe {
            let c_path = CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
            binding = ffi::binding_load(dims, c_path.as_ptr() as _);
        }
        Self {
            binding,
            options_common,
            options_algo,
        }
    }
    pub fn save(&mut self, path: impl AsRef<Path>) {
        unsafe {
            let c_path = CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
            ffi::binding_save(self.binding, c_path.as_ptr() as _);
        }
    }
    pub fn insert(&mut self, vector: &[Scalar], p: u64) {
        unsafe {
            assert!(vector.len() == self.options_common.dims as usize);
            let vector = vector.iter().map(|x| *x as f32).collect::<Vec<f32>>();
            ffi::binding_insert(self.binding, vector.as_ptr() as _, p as _);
        }
    }
    pub fn search(&mut self, vector: &[Scalar], k: usize) -> Vec<u64> {
        unsafe {
            assert!(vector.len() == self.options_common.dims as usize);
            let vector = vector.iter().map(|x| *x as f32).collect::<Vec<f32>>();
            let result = ffi::binding_search(self.binding, vector.as_ptr() as _, k).into_iter();
            result.map(|x| x as _).collect()
        }
    }
}

impl Drop for AlgoHnsw {
    fn drop(&mut self) {
        unsafe {
            ffi::binding_delete(self.binding);
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct AlgoHnswOptions {}
