use crate::prelude::Scalar;

// "Tcalar" is "Total-ordering scalar".
#[derive(Debug, Clone, Copy, Default)]
pub struct Tcalar(pub Scalar);

impl PartialEq for Tcalar {
    fn eq(&self, other: &Self) -> bool {
        use std::cmp::Ordering;
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Tcalar {}

impl PartialOrd for Tcalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tcalar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
