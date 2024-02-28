use std::path::Path;

pub trait Module {
    type Options: Sized;
    type External: Sized;
    fn create(path: &Path, options: Self::Options, external: Self::External);
    fn open(path: &Path, external: Self::External);
    fn path(&self) -> &Path;
}
