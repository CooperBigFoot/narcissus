//! Mini-batch K-means algorithm implementation.
//!
//! Provides a faster variant of K-means that performs centroid updates on
//! random subsets of the data at each iteration. Centroid positions are
//! updated using a decaying per-centroid learning rate, which approximates
//! the full-batch update at a fraction of the cost for large datasets.
//!
//! A final full-pass assignment step is performed after the iterative phase
//! to compute accurate cluster labels and inertia.

use std::cmp::Ordering;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use tracing::{debug, info, instrument};

use narcissus_dtw::{BandConstraint, Dtw, TimeSeries};

use crate::config::MiniBatchConfig;
use crate::error::ClusterError;
use crate::inertia::Inertia;
use crate::init::kmeans_plus_plus;
use crate::label::ClusterLabel;
use crate::result::KMeansResult;

// ── Internal run result ───────────────────────────────────────────────────────

/// Result of a single mini-batch K-means restart.
struct SingleRun {
    assignments: Vec<ClusterLabel>,
    centroids: Vec<TimeSeries>,
    inertia: Inertia,
    converged: bool,
    iterations: usize,
}

// ── Fisher-Yates sampling ─────────────────────────────────────────────────────

/// Sample `batch_size` distinct indices from `0..n` without replacement using
/// a partial Fisher-Yates shuffle.
///
/// When `batch_size >= n`, all `n` indices are returned in a shuffled order.
fn sample_without_replacement(n: usize, batch_size: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let actual = batch_size.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    // Perform a partial Fisher-Yates shuffle: swap each of the first `actual`
    // positions with a randomly chosen position at or after it.
    for i in 0..actual {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    indices.truncate(actual);
    indices
}

// ── run_once_minibatch ────────────────────────────────────────────────────────

/// Run a single mini-batch K-means restart seeded with `seed`.
///
/// Centroids are initialized via K-means++ then refined over `config.max_iter`
/// iterations, each operating on a random batch of `config.batch_size` series.
/// A final full-pass assignment determines the returned labels and inertia.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`ClusterError::TooFewSeries`] | Internal invariant; should not fire if caller validates |
#[instrument(skip(series, config), fields(k = config.k, seed))]
fn run_once_minibatch(
    series: &[TimeSeries],
    config: &MiniBatchConfig,
    seed: u64,
) -> Result<SingleRun, ClusterError> {
    let n = series.len();
    let series_len = series[0].len();
    let k = config.k;

    let dtw = match config.constraint {
        BandConstraint::Unconstrained => Dtw::unconstrained(),
        BandConstraint::SakoeChibaRadius(r) => Dtw::with_sakoe_chiba(r),
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // ── Initialization ──────────────────────────────────────────────────────

    // Initialize centroid indices via K-means++.
    let init_indices = kmeans_plus_plus(series, k, &dtw, &mut rng);

    // Work with raw Vec<Vec<f64>> for centroids during the iterative phase
    // so we can perform in-place weighted updates without TimeSeries immutability
    // constraints.
    let mut raw_centroids: Vec<Vec<f64>> = init_indices
        .iter()
        .map(|&i| series[i].as_ref().to_vec())
        .collect();

    // Per-centroid counts: how many batch points have been assigned to each
    // centroid so far. Used to compute the decaying learning rate.
    let mut counts: Vec<usize> = vec![0usize; k];

    let mut converged = false;
    let mut iterations = 0usize;

    // ── Mini-batch iterations ───────────────────────────────────────────────

    for iteration in 0..config.max_iter {
        iterations = iteration + 1;

        // Sample a batch of indices without replacement.
        let batch_size = config.batch_size.min(n);
        let batch_indices = sample_without_replacement(n, batch_size, &mut rng);

        // ── Assign batch members to nearest centroid ──────────────────────
        // We convert raw_centroids to TimeSeries temporarily for DTW, but avoid
        // cloning the full centroid vec on every iteration by building views.
        let centroid_ts: Vec<TimeSeries> = raw_centroids
            .iter()
            .map(|c| TimeSeries::new(c.clone()).expect("centroid values are finite by construction"))
            .collect();

        // Compute per-batch-item assignments (sequential: batch is small).
        let batch_assignments: Vec<usize> = batch_indices
            .iter()
            .map(|&i| {
                let view = series[i].as_view();
                let mut best_label = 0usize;
                let mut best_dist = f64::INFINITY;
                for (c_idx, centroid) in centroid_ts.iter().enumerate() {
                    let d = dtw.distance(view, centroid.as_view()).value();
                    if d < best_dist {
                        best_dist = d;
                        best_label = c_idx;
                    }
                }
                best_label
            })
            .collect();

        // ── Update centroids with decaying learning rate ───────────────────
        // Snapshot centroid positions before the batch updates so we can
        // compute how far each centroid actually moved this iteration.
        let prev_raw_centroids: Vec<Vec<f64>> = raw_centroids.clone();

        for (&series_idx, &centroid_idx) in batch_indices.iter().zip(batch_assignments.iter()) {
            counts[centroid_idx] += 1;
            let lr = 1.0 / counts[centroid_idx] as f64;

            let series_data = series[series_idx].as_ref();
            let centroid = &mut raw_centroids[centroid_idx];

            for d in 0..series_len {
                let old = centroid[d];
                centroid[d] = (1.0 - lr) * old + lr * series_data[d];
            }
        }

        // ── Convergence check ─────────────────────────────────────────────
        // Measure the maximum L2 displacement of any centroid this iteration.
        let max_centroid_shift: f64 = raw_centroids
            .iter()
            .zip(prev_raw_centroids.iter())
            .map(|(new_c, old_c)| {
                new_c
                    .iter()
                    .zip(old_c.iter())
                    .map(|(n, o)| (n - o).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0f64, f64::max);

        debug!(
            iteration,
            max_centroid_shift,
            batch_size = batch_indices.len(),
            "mini-batch iteration complete"
        );

        if max_centroid_shift < config.tol {
            converged = true;
            debug!(iteration, "converged");
            break;
        }
    }

    // ── Final full-pass assignment ──────────────────────────────────────────

    // Convert raw centroids back to TimeSeries for the final assignment.
    let final_centroids: Vec<TimeSeries> = raw_centroids
        .into_iter()
        .map(|c| TimeSeries::new(c).expect("centroid values are finite by construction"))
        .collect();

    // Parallel full-pass assignment to get accurate labels and inertia.
    let assign_results: Vec<(ClusterLabel, f64)> = series
        .par_iter()
        .map(|s| {
            let view = s.as_view();
            let mut best_label = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c_idx, centroid) in final_centroids.iter().enumerate() {
                let d = dtw.distance(view, centroid.as_view()).value();
                if d < best_dist {
                    best_dist = d;
                    best_label = c_idx;
                }
            }
            (ClusterLabel::new(best_label), best_dist)
        })
        .collect();

    let inertia_value: f64 = assign_results.iter().map(|(_, d)| d.powi(2)).sum();
    let assignments: Vec<ClusterLabel> =
        assign_results.into_iter().map(|(label, _)| label).collect();

    info!(
        seed,
        iterations,
        inertia = inertia_value,
        converged,
        "mini-batch single restart complete"
    );

    Ok(SingleRun {
        assignments,
        centroids: final_centroids,
        inertia: Inertia::new(inertia_value),
        converged,
        iterations,
    })
}

// ── multi_restart_minibatch ───────────────────────────────────────────────────

/// Run `config.n_init` independent mini-batch K-means restarts and return the
/// best result by inertia.
///
/// Restarts are executed in parallel with rayon. Sub-seeds are derived
/// deterministically from `config.seed` so the overall computation is
/// reproducible.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`ClusterError::TooFewSeries`] | `series.len() < k` (validated by caller) |
#[instrument(skip(series, config), fields(k = config.k, n_init = config.n_init))]
pub(crate) fn multi_restart_minibatch(
    series: &[TimeSeries],
    config: &MiniBatchConfig,
) -> Result<KMeansResult, ClusterError> {
    // Derive per-restart seeds deterministically from the master seed.
    let mut master_rng = ChaCha8Rng::seed_from_u64(config.seed);
    let seeds: Vec<u64> = (0..config.n_init).map(|_| master_rng.r#gen()).collect();

    let n_init = seeds.len();

    // Run all restarts in parallel, collecting results (including errors).
    let results: Vec<Result<SingleRun, ClusterError>> = seeds
        .into_par_iter()
        .map(|seed| run_once_minibatch(series, config, seed))
        .collect();

    // Propagate any error encountered; pick the best successful run.
    let mut best: Option<SingleRun> = None;
    let mut n_ok = 0usize;

    for result in results {
        let run = result?;
        n_ok += 1;
        best = Some(match best {
            None => run,
            Some(prev) => {
                if run.inertia.total_cmp(&prev.inertia) == Ordering::Less {
                    run
                } else {
                    prev
                }
            }
        });
    }

    // `best` is always Some here because n_init >= 1 and all results succeeded.
    let best = best.expect("at least one restart must succeed");

    info!(
        k = config.k,
        n_init,
        n_ok,
        best_inertia = best.inertia.value(),
        "mini-batch multi-restart complete"
    );

    Ok(KMeansResult {
        assignments: best.assignments,
        centroids: best.centroids,
        inertia: best.inertia,
        converged: best.converged,
        iterations: best.iterations,
        n_init_used: n_ok,
    })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use narcissus_dtw::{BandConstraint, TimeSeries};

    use super::multi_restart_minibatch;
    use crate::config::MiniBatchConfig;
    use crate::error::ClusterError;

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

    // Test 1: batch_size larger than n should NOT error; it clamps to n.
    #[test]
    fn batch_too_large_clamps_to_n() {
        let series = archetype_a(); // 9 series
        let cfg = MiniBatchConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_batch_size(1000) // larger than n=9 → clamps to 9
            .with_n_init(1)
            .with_seed(42);
        // Should succeed; batch_size is clamped inside run_once_minibatch.
        let result = multi_restart_minibatch(&series, &cfg);
        assert!(
            result.is_ok(),
            "batch_size > n should succeed (clamp to n), got {:?}",
            result.err()
        );
    }

    // Test 2: three well-separated clusters should be recovered correctly.
    #[test]
    fn three_clusters_quality() {
        let series = archetype_a();
        let cfg = MiniBatchConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_batch_size(6)
            .with_n_init(5)
            .with_max_iter(300)
            .with_seed(42);
        let result = multi_restart_minibatch(&series, &cfg).unwrap();

        assert_eq!(result.centroids.len(), 3, "should have 3 centroids");
        assert_eq!(result.assignments.len(), 9, "should have 9 assignments");

        // Each of the three archetype groups (indices 0-2, 3-5, 6-8) must map
        // to exactly one distinct cluster label.
        let archetype_group = |idx: usize| -> usize {
            match idx {
                0..=2 => 0,
                3..=5 => 1,
                6..=8 => 2,
                _ => panic!("unexpected index {idx}"),
            }
        };

        let mut label_to_archetype: Vec<Option<usize>> = vec![None; 3];
        for (i, label) in result.assignments.iter().enumerate() {
            let ag = archetype_group(i);
            let c = label.index();
            match label_to_archetype[c] {
                None => label_to_archetype[c] = Some(ag),
                Some(prev) => assert_eq!(
                    prev, ag,
                    "cluster {c} contains series from different archetype groups"
                ),
            }
        }

        let mut seen: Vec<usize> = label_to_archetype
            .into_iter()
            .map(|o| o.expect("every cluster must be occupied"))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2], "each archetype group must form its own cluster");
    }

    // Test 3: same seed → same assignments and inertia.
    #[test]
    fn deterministic() {
        let series = archetype_a();
        let cfg = MiniBatchConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_batch_size(6)
            .with_n_init(3)
            .with_seed(77);

        let r1 = multi_restart_minibatch(&series, &cfg).unwrap();
        let r2 = multi_restart_minibatch(&series, &cfg).unwrap();

        let labels1: Vec<usize> = r1.assignments.iter().map(|l| l.index()).collect();
        let labels2: Vec<usize> = r2.assignments.iter().map(|l| l.index()).collect();

        assert_eq!(labels1, labels2, "results must be deterministic with same seed");
        assert_eq!(
            r1.inertia.value(),
            r2.inertia.value(),
            "inertia must be deterministic with same seed"
        );
    }

    // Test 4: k=1 should assign all series to cluster 0.
    #[test]
    fn k_one_trivial() {
        let series = archetype_a();
        let cfg = MiniBatchConfig::new(1, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(1)
            .with_seed(0);
        let result = multi_restart_minibatch(&series, &cfg).unwrap();

        assert_eq!(result.centroids.len(), 1, "should have exactly 1 centroid");
        assert!(
            result.assignments.iter().all(|l| l.index() == 0),
            "all assignments must be cluster 0"
        );
        assert!(result.inertia.value() > 0.0, "inertia must be positive");
    }

    // Test 5: should converge within max_iter on well-separated data.
    #[test]
    fn convergence_on_easy_data() {
        let series = archetype_a();
        let cfg = MiniBatchConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_batch_size(9) // full batch
            .with_n_init(5)
            .with_max_iter(500)
            .with_tol(1e-4)
            .with_seed(42);
        let result = multi_restart_minibatch(&series, &cfg).unwrap();

        assert!(
            result.converged,
            "should converge on well-separated data within 500 iterations"
        );
    }

    // Test 6: TooFewSeries error when n < k.
    #[test]
    fn too_few_series_error() {
        let series = vec![
            TimeSeries::new(vec![1.0, 2.0]).unwrap(),
            TimeSeries::new(vec![3.0, 4.0]).unwrap(),
        ];
        let cfg = MiniBatchConfig::new(5, BandConstraint::Unconstrained).unwrap();
        let result = cfg.fit(&series);
        assert!(
            matches!(result, Err(ClusterError::TooFewSeries { n_series: 2, k: 5 })),
            "expected TooFewSeries, got {:?}",
            result.err()
        );
    }
}
