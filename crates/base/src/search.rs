use crate::scalar::F32;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::num::ParseIntError;
use std::str::FromStr;

#[derive(
    Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Payload {
    pub pointer: Pointer,
    pub time: u64,
}

impl Payload {
    pub fn new(pointer: Pointer, time: u64) -> Self {
        Self { pointer, time }
    }
    pub fn pointer(self) -> Pointer {
        self.pointer
    }
    pub fn time(self) -> u64 {
        self.time
    }
}

unsafe impl bytemuck::Zeroable for Payload {}
unsafe impl bytemuck::Pod for Payload {}

pub trait Filter: Clone {
    fn check(&mut self, payload: Payload) -> bool;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Element {
    pub distance: F32,
    pub payload: Payload,
}

#[derive(
    Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Handle {
    pub newtype: u32,
}

impl Handle {
    pub fn as_u32(self) -> u32 {
        self.newtype
    }
}

impl Display for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_u32())
    }
}

impl FromStr for Handle {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Handle {
            newtype: u32::from_str(s)?,
        })
    }
}

#[derive(
    Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Pointer {
    pub newtype: u64,
}
