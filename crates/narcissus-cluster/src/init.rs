//! K-means++ initialization (private module).
//!
//! Implements the K-means++ seeding algorithm, which selects initial centroid
//! indices with a probability proportional to the squared DTW distance from
//! each candidate to the nearest already-chosen centroid. This improves
//! convergence speed and final clustering quality compared to uniform random
//! initialization.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use narcissus_dtw::{Dtw, TimeSeries};

/// Select `k` initial centroid indices from `series` using the K-means++
/// seeding algorithm.
///
/// Returns a `Vec<usize>` of length `k` containing distinct indices into
/// `series`. The first centroid is chosen uniformly at random; each subsequent
/// centroid is drawn with probability proportional to the squared DTW distance
/// from that series to the nearest already-chosen centroid.
///
/// The per-series distance computation is parallelized with rayon. The weighted
/// sampling step is sequential because it requires a mutable reference to `rng`.
///
/// # Panics
///
/// Panics in debug mode if `k == 0` or `k > series.len()`.
#[must_use]
pub(crate) fn kmeans_plus_plus(
    series: &[TimeSeries],
    k: usize,
    dtw: &Dtw,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let n = series.len();
    debug_assert!(k > 0, "k must be at least 1");
    debug_assert!(k <= n, "k must not exceed the number of series");

    // Chosen centroid indices accumulated across iterations.
    let mut chosen: Vec<usize> = Vec::with_capacity(k);

    // Step 1: pick the first centroid uniformly at random.
    let first = rng.gen_range(0..n);
    chosen.push(first);

    // Step 2: iteratively pick c = 1..k centroids.
    for _ in 1..k {
        // 2a. For each series i, compute D(i) = min distance to any chosen centroid.
        // Parallelized over the n series; the inner loop over `chosen` is sequential
        // because `chosen.len()` is small (at most k-1 << n in typical use).
        let weights: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                // 2c. Already-chosen series get weight 0 so they are never re-selected.
                if chosen.contains(&i) {
                    return 0.0;
                }

                // 2a. Minimum DTW distance to any existing centroid.
                let min_dist = chosen
                    .iter()
                    .map(|&j| dtw.distance(series[i].as_view(), series[j].as_view()).value())
                    .fold(f64::INFINITY, f64::min);

                // 2b. Weight is the squared distance.
                min_dist.powi(2)
            })
            .collect();

        // 2d. Weighted random sampling (sequential — requires mutable rng).
        let total_weight: f64 = weights.iter().sum();

        if total_weight == 0.0 {
            // All remaining series are at distance 0 from existing centroids
            // (e.g., duplicate series). Fall back to any unchosen index.
            let fallback = (0..n)
                .find(|i| !chosen.contains(i))
                .expect("k <= n guarantees an unchosen index exists");
            chosen.push(fallback);
            continue;
        }

        // Draw a random threshold in [0, total_weight).
        let threshold: f64 = rng.gen_range(0.0..total_weight);

        // Walk the cumulative sum to find the selected index.
        let mut cumsum = 0.0;
        let mut selected = n - 1; // safe default: last index
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                selected = i;
                break;
            }
        }

        chosen.push(selected);
    }

    chosen
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    use narcissus_dtw::{Dtw, TimeSeries};

    use super::kmeans_plus_plus;

    /// Nine series in three tight clusters near 0, 5, and 10.
    fn archetype_a() -> Vec<TimeSeries> {
        vec![
            TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![0.1, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![0.0, 0.1, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![5.0, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![5.1, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![5.0, 5.1, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0]).unwrap(),
            TimeSeries::new(vec![10.1, 10.0, 10.0, 10.0]).unwrap(),
            TimeSeries::new(vec![10.0, 10.1, 10.0, 10.0]).unwrap(),
        ]
    }

    #[test]
    fn returns_k_distinct_indices() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let indices = kmeans_plus_plus(&series, 3, &dtw, &mut rng);

        assert_eq!(indices.len(), 3, "should return exactly k indices");

        // All indices must be in-bounds.
        for &idx in &indices {
            assert!(idx < 9, "index {idx} out of range 0..9");
        }

        // All indices must be distinct.
        let mut dedup = indices.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), 3, "indices must be distinct");
    }

    #[test]
    fn k_equals_n_returns_all() {
        let series = vec![
            TimeSeries::new(vec![1.0, 2.0]).unwrap(),
            TimeSeries::new(vec![3.0, 4.0]).unwrap(),
            TimeSeries::new(vec![5.0, 6.0]).unwrap(),
        ];
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let mut indices = kmeans_plus_plus(&series, 3, &dtw, &mut rng);
        indices.sort_unstable();

        assert_eq!(indices, vec![0, 1, 2], "k=n should cover all 3 indices");
    }

    #[test]
    fn k_one_returns_single() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        let indices = kmeans_plus_plus(&series, 1, &dtw, &mut rng);

        assert_eq!(indices.len(), 1, "k=1 should return exactly one index");
        assert!(indices[0] < 9, "index out of range");
    }

    #[test]
    fn deterministic_with_same_seed() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let indices1 = kmeans_plus_plus(&series, 3, &dtw, &mut rng1);
        let indices2 = kmeans_plus_plus(&series, 3, &dtw, &mut rng2);

        assert_eq!(indices1, indices2, "same seed must produce identical results");
    }

    #[test]
    fn prefers_distant_centroids() {
        // Archetype A has three well-separated clusters:
        //   Group 0: indices 0, 1, 2  (values near 0)
        //   Group 1: indices 3, 4, 5  (values near 5)
        //   Group 2: indices 6, 7, 8  (values near 10)
        //
        // K-means++ should strongly prefer one centroid per group.
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let indices = kmeans_plus_plus(&series, 3, &dtw, &mut rng);

        // Map each chosen index to its group (0, 1, or 2).
        let group = |idx: usize| -> usize {
            match idx {
                0..=2 => 0,
                3..=5 => 1,
                6..=8 => 2,
                _ => panic!("unexpected index {idx}"),
            }
        };

        let groups: Vec<usize> = indices.iter().map(|&i| group(i)).collect();
        let mut sorted_groups = groups.clone();
        sorted_groups.sort_unstable();
        sorted_groups.dedup();

        assert_eq!(
            sorted_groups.len(),
            3,
            "expected one centroid from each group, got groups {groups:?} for indices {indices:?}"
        );
    }
}
