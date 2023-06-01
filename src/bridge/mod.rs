#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("vectors/src/bridge/bindings.cpp");

        pub type Binding;

        pub unsafe fn binding_save(sel: *mut Binding, location: *const u8);
        pub unsafe fn binding_insert(sel: *mut Binding, data: *const u8, label: usize);
        pub unsafe fn binding_search(sel: *mut Binding, data: *const u8, k: usize) -> Vec<usize>;

        pub unsafe fn binding_new(dim: usize) -> *mut Binding;
        pub unsafe fn binding_load(dim: usize, location: *const u8) -> *mut Binding;
        pub unsafe fn binding_delete(sel: *mut Binding);
    }
}
