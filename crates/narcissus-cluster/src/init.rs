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

use narcissus_dtw::{DistanceMatrix, Dtw, TimeSeries};

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

/// Select `k` initial centroid indices using K-means++ with a precomputed distance matrix.
///
/// Identical algorithm to [`kmeans_plus_plus`], but looks up pairwise distances from
/// `matrix` instead of computing DTW on the fly. This avoids redundant distance
/// computations when the same pairwise matrix is reused across multiple restarts.
///
/// # Panics
///
/// Panics in debug mode if `k == 0`, `k > n_series`, or `n_series != matrix.len()`.
#[must_use]
pub(crate) fn kmeans_plus_plus_cached(
    n_series: usize,
    k: usize,
    matrix: &DistanceMatrix,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    debug_assert!(k > 0, "k must be at least 1");
    debug_assert!(k <= n_series, "k must not exceed the number of series");
    debug_assert_eq!(
        n_series,
        matrix.len(),
        "n_series must match matrix dimension"
    );

    let mut chosen: Vec<usize> = Vec::with_capacity(k);

    // Step 1: pick the first centroid uniformly at random.
    let first = rng.gen_range(0..n_series);
    chosen.push(first);

    // Step 2: iteratively pick c = 1..k centroids.
    for _ in 1..k {
        // 2a. For each series i, compute D(i) = min distance to any chosen centroid.
        let weights: Vec<f64> = (0..n_series)
            .into_par_iter()
            .map(|i| {
                if chosen.contains(&i) {
                    return 0.0;
                }

                let min_dist = chosen
                    .iter()
                    .map(|&j| matrix.get(i, j).value())
                    .fold(f64::INFINITY, f64::min);

                min_dist.powi(2)
            })
            .collect();

        // 2b. Weighted random sampling (sequential — requires mutable rng).
        let total_weight: f64 = weights.iter().sum();

        if total_weight == 0.0 {
            let fallback = (0..n_series)
                .find(|i| !chosen.contains(i))
                .expect("k <= n guarantees an unchosen index exists");
            chosen.push(fallback);
            continue;
        }

        let threshold: f64 = rng.gen_range(0.0..total_weight);

        let mut cumsum = 0.0;
        let mut selected = n_series - 1;
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

/// Select `k` initial centroid indices using K-Means‖ (K-Means Parallel) oversampled
/// D²-sampling (Bahmani et al. 2012, "Scalable K-Means++").
///
/// The algorithm runs O(log k) oversampling rounds. In each round, every series is
/// sampled independently with probability `p = oversample_factor * D(x)^2 / cost`
/// where `D(x)` is the distance from series `x` to the nearest current center and
/// `cost` is the sum of all squared distances. After oversampling, the (potentially
/// many) candidate centers are reduced to exactly `k` by running weighted K-means++
/// on the candidates, weighting each candidate by the number of series closest to it.
///
/// # Panics
///
/// Panics in debug mode if `k == 0` or `k > series.len()`.
#[must_use]
pub(crate) fn kmeans_parallel_init(
    series: &[TimeSeries],
    k: usize,
    oversample_factor: f64,
    dtw: &Dtw,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    use rand::Rng as _;
    use tracing::debug;

    let n = series.len();
    debug_assert!(k > 0, "k must be at least 1");
    debug_assert!(k <= n, "k must not exceed the number of series");

    // Step 1: pick the first center uniformly at random.
    let first = rng.gen_range(0..n);
    let mut centers: Vec<usize> = vec![first];

    // Number of oversampling rounds: ceil(log2(k)).max(1).
    let n_rounds = ((k as f64).log2().ceil() as usize).max(1);

    debug!(k, n_rounds, oversample_factor, "kmeans_parallel_init starting");

    // Steps 2: oversampling rounds.
    for _round in 0..n_rounds {
        // 2a. Compute D(x)^2 for each series (min squared distance to any current center).
        let sq_dists: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let min_dist = centers
                    .iter()
                    .map(|&j| dtw.distance(series[i].as_view(), series[j].as_view()).value())
                    .fold(f64::INFINITY, f64::min);
                min_dist.powi(2)
            })
            .collect();

        let cost: f64 = sq_dists.iter().sum();

        // If all points are already on a center, stop early.
        if cost == 0.0 {
            break;
        }

        // 2b. Sample each point independently with probability p = oversample_factor * D(x)^2 / cost.
        for (i, &sq_d) in sq_dists.iter().enumerate() {
            let p = (oversample_factor * sq_d / cost).min(1.0);
            let u: f64 = rng.gen_range(0.0..1.0);
            if u < p {
                centers.push(i);
            }
        }
    }

    // De-duplicate centers (keep unique indices).
    centers.sort_unstable();
    centers.dedup();

    debug!(n_candidates = centers.len(), k, "oversampling complete, reducing candidates");

    // Step 3: reduce to exactly k centers via weighted K-means++.
    // If we somehow ended up with fewer than k candidates (e.g., n == k),
    // fall back to including all series as candidates.
    if centers.len() < k {
        // Not enough candidates — supplement with remaining series indices.
        for i in 0..n {
            if !centers.contains(&i) {
                centers.push(i);
            }
            if centers.len() >= k {
                break;
            }
        }
        centers.sort_unstable();
        centers.dedup();
    }

    // If we have exactly k candidates, return them directly.
    if centers.len() == k {
        return centers;
    }

    // Compute weights: for each candidate, count how many series are closest to it.
    // This is the importance weight used in the weighted K-means++ reduction.
    let n_candidates = centers.len();
    let weights: Vec<f64> = {
        // For each series, find the closest candidate center.
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                centers
                    .iter()
                    .enumerate()
                    .map(|(ci, &j)| {
                        (ci, dtw.distance(series[i].as_view(), series[j].as_view()).value())
                    })
                    .min_by(|(_, da), (_, db)| da.total_cmp(db))
                    .map(|(ci, _)| ci)
                    .unwrap_or(0)
            })
            .collect();

        let mut w = vec![0.0f64; n_candidates];
        for &ci in &assignments {
            w[ci] += 1.0;
        }
        w
    };

    // Weighted K-means++ on the candidates.
    // chosen_candidate_indices is a Vec of positions into `centers`.
    let mut chosen_pos: Vec<usize> = Vec::with_capacity(k);

    // Pick the first candidate proportional to its weight.
    {
        let total_w: f64 = weights.iter().sum();
        let threshold: f64 = rng.gen_range(0.0..total_w.max(f64::EPSILON));
        let mut cumsum = 0.0;
        let mut sel = n_candidates - 1;
        for (ci, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                sel = ci;
                break;
            }
        }
        chosen_pos.push(sel);
    }

    // Pick subsequent candidates via weighted D²-sampling on candidates.
    for _ in 1..k {
        // For each candidate, compute min squared DTW distance to any chosen candidate,
        // scaled by the candidate's importance weight.
        let wts: Vec<f64> = (0..n_candidates)
            .map(|ci| {
                if chosen_pos.contains(&ci) {
                    return 0.0;
                }
                let min_dist = chosen_pos
                    .iter()
                    .map(|&cj| {
                        dtw.distance(series[centers[ci]].as_view(), series[centers[cj]].as_view())
                            .value()
                    })
                    .fold(f64::INFINITY, f64::min);
                weights[ci] * min_dist.powi(2)
            })
            .collect();

        let total_wt: f64 = wts.iter().sum();

        if total_wt == 0.0 {
            // Fall back to any unchosen candidate.
            let fallback = (0..n_candidates)
                .find(|ci| !chosen_pos.contains(ci))
                .expect("k <= n_candidates guarantees an unchosen candidate exists");
            chosen_pos.push(fallback);
            continue;
        }

        let threshold: f64 = rng.gen_range(0.0..total_wt);
        let mut cumsum = 0.0;
        let mut selected = n_candidates - 1;
        for (ci, &w) in wts.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                selected = ci;
                break;
            }
        }
        chosen_pos.push(selected);
    }

    // Map chosen candidate positions back to original series indices.
    let result: Vec<usize> = chosen_pos.iter().map(|&ci| centers[ci]).collect();

    debug!(k, "kmeans_parallel_init complete");
    result
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    use narcissus_dtw::{Dtw, TimeSeries};

    use super::{kmeans_parallel_init, kmeans_plus_plus, kmeans_plus_plus_cached};

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

    #[test]
    fn cached_init_matches_uncached() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let matrix = dtw.pairwise(&series);

        let seed = 42u64;
        let k = 3;

        let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
        let uncached = kmeans_plus_plus(&series, k, &dtw, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let cached = kmeans_plus_plus_cached(series.len(), k, &matrix, &mut rng2);

        assert_eq!(
            uncached, cached,
            "cached init must produce identical indices as uncached with same seed"
        );
    }

    // ── kmeans_parallel_init tests ────────────────────────────────────────────

    #[test]
    fn parallel_init_k_distinct() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let indices = kmeans_parallel_init(&series, 3, 2.0, &dtw, &mut rng);

        assert_eq!(indices.len(), 3, "should return exactly k=3 indices");

        for &idx in &indices {
            assert!(idx < 9, "index {idx} out of range 0..9");
        }

        let mut dedup = indices.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), 3, "all returned indices must be distinct");
    }

    #[test]
    fn parallel_init_deterministic() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();

        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        let mut rng2 = ChaCha8Rng::seed_from_u64(77);

        let indices1 = kmeans_parallel_init(&series, 3, 2.0, &dtw, &mut rng1);
        let indices2 = kmeans_parallel_init(&series, 3, 2.0, &dtw, &mut rng2);

        assert_eq!(indices1, indices2, "same seed must produce identical results");
    }

    #[test]
    fn parallel_init_prefers_distant_centroids() {
        // Archetype A has three well-separated clusters:
        //   Group 0: indices 0, 1, 2  (values near 0)
        //   Group 1: indices 3, 4, 5  (values near 5)
        //   Group 2: indices 6, 7, 8  (values near 10)
        //
        // K-Means‖ should pick one centroid from each group.
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let indices = kmeans_parallel_init(&series, 3, 2.0, &dtw, &mut rng);

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

    #[test]
    fn parallel_init_k_one() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(5);

        let indices = kmeans_parallel_init(&series, 1, 2.0, &dtw, &mut rng);

        assert_eq!(indices.len(), 1, "k=1 should return exactly one index");
        assert!(indices[0] < 9, "index {} out of range 0..9", indices[0]);
    }

    #[test]
    fn parallel_init_k_equals_n() {
        let series = vec![
            TimeSeries::new(vec![1.0, 2.0]).unwrap(),
            TimeSeries::new(vec![3.0, 4.0]).unwrap(),
            TimeSeries::new(vec![5.0, 6.0]).unwrap(),
        ];
        let dtw = Dtw::unconstrained();
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let mut indices = kmeans_parallel_init(&series, 3, 2.0, &dtw, &mut rng);
        indices.sort_unstable();

        assert_eq!(indices, vec![0, 1, 2], "k=n should return all 3 indices");
    }
}
