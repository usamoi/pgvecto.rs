use crate::utils::tcalar::Tcalar;
use crate::utils::vec2::Vec2;
use crate::prelude::{Distance, Scalar};
use rand::seq::index::sample;
use rand::seq::SliceRandom;
use rand::Rng;

pub fn kmeans(distance: Distance, dims: u16, v: &Vec2, k: usize) -> Vec2 {
    let eps = 1.0 / 1024.0;
    let iter = 25;
    let min_points_per_centroid = 39;
    let max_points_per_centroid = 256;
    let mut rand = rand::thread_rng();
    let f;

    if v.len() > k * max_points_per_centroid {
        f = sample(&mut rand, v.len(), k * max_points_per_centroid).into_vec();
    } else if v.len() < k * min_points_per_centroid {
        f = sample(&mut rand, v.len(), v.len()).into_vec();
        log::warn!("Provided too few training points");
    } else {
        f = sample(&mut rand, v.len(), v.len()).into_vec();
    }

    let n = f.len();
    if n <= k {
        let mut centroids = Vec2::new(dims, k);
        for i in 0..k {
            centroids[i].copy_from_slice(&v[i % n]);
        }
        return centroids;
    }

    let mut centroids = Vec2::new(dims, k);
    let mut assign = vec![0usize; n];
    let mut hassign = vec![0.0; k];

    for (i, &index) in f.choose_multiple(&mut rand, k).enumerate() {
        centroids[i].copy_from_slice(&v[index]);
    }

    for _ in 0..iter {
        for i in 0..n {
            let mut result = (Tcalar(Scalar::INFINITY), 0);
            for j in 0..k {
                let dis = Tcalar(distance.distance(&centroids[j], &v[f[i]]));
                result = std::cmp::min(result, (dis, j));
            }
            assign[i] = result.1;
        }
        centroids.fill(0.0);
        hassign.fill(0.0);
        for i in 0..n {
            hassign[assign[i]] += 1.0;
            for j in 0..dims as usize {
                centroids[assign[i]][j] += v[f[i]][j];
            }
        }
        for i in 0..k {
            if hassign[i] == 0.0 {
                continue;
            }
            let norm = 1.0 / hassign[i];
            for j in 0..dims as usize {
                centroids[i][j] *= norm;
            }
        }
        for i in 0..k {
            if hassign[i] != 0.0 {
                continue;
            }
            let mut j = 0;
            loop {
                let p = (hassign[j] - 1.0) / (n - k) as Scalar;
                let r = rand.gen_range(0.0..1.0);
                if r < p {
                    break;
                }
                j = (j + 1) % k;
            }
            centroids.copy_within(j, i);
            for p in 0..dims as usize {
                if p % 2 == 0 {
                    centroids[i][p] *= 1.0 + eps;
                    centroids[j][p] *= 1.0 - eps;
                } else {
                    centroids[i][p] *= 1.0 - eps;
                    centroids[j][p] *= 1.0 + eps;
                }
            }
            hassign[i] = hassign[i] / 2.0;
            hassign[j] = hassign[j] - hassign[i];
        }
    }

    centroids
}
