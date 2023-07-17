pub struct Visited {
    version: usize,
    data: Box<[usize]>,
}

impl Visited {
    pub fn new(capacity: usize) -> Self {
        Self {
            version: 0,
            data: unsafe { Box::new_zeroed_slice(capacity).assume_init() },
        }
    }
    pub fn new_version(&mut self) -> VisitedVersion<'_> {
        assert_ne!(self.version, usize::MAX);
        self.version += 1;
        VisitedVersion { inner: self }
    }
}

pub struct VisitedVersion<'a> {
    inner: &'a mut Visited,
}

impl<'a> VisitedVersion<'a> {
    pub fn test(&mut self, i: usize) -> bool {
        self.inner.data[i] == self.inner.version
    }
    pub fn set(&mut self, i: usize) {
        self.inner.data[i] = self.inner.version;
    }
}
