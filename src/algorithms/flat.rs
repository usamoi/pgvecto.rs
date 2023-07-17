use crate::algorithms::Vectors;
use crate::prelude::Algorithm;
use crate::prelude::Distance;
use crate::prelude::Options;
use crate::prelude::Scalar;
use crate::utils::tcalar::Tcalar;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlatOptions {}

pub struct Flat {
    distance: Distance,
    vectors: Arc<Vectors>,
}

impl Algorithm for Flat {
    type Options = FlatOptions;

    fn build(options: Options, vectors: Arc<Vectors>, _: usize) -> anyhow::Result<(Self, Vec<u8>)> {
        Ok((
            Self {
                distance: options.distance,
                vectors,
            },
            Vec::new(),
        ))
    }

    fn load(options: Options, vectors: Arc<Vectors>, _: Vec<u8>) -> anyhow::Result<Self> {
        Ok(Self {
            distance: options.distance,
            vectors,
        })
    }

    fn insert(&self, _: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn search(&self, (vector, k): (Box<[Scalar]>, usize)) -> anyhow::Result<Vec<(Scalar, u64)>> {
        let mut answer = BinaryHeap::<Reverse<(Tcalar, u64)>>::new();
        for i in 0..self.vectors.len() {
            let v = self.vectors.get_vector(i).as_ref();
            let p = self.vectors.get_data(i);
            let d = self.distance.distance(&vector, &v);
            answer.push(Reverse((Tcalar(d), p)));
            if answer.len() > k {
                answer.pop();
            }
        }
        let mut answer = answer
            .into_iter()
            .map(|x| (x.0 .0 .0, x.0 .1))
            .collect::<Vec<_>>();
        answer.sort_by_key(|x| Tcalar(x.0));
        Ok(answer)
    }
}
