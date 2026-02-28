//! Core K-means algorithm implementation.
//!
//! Provides the assign/update EM loop, multi-restart orchestration, and the
//! sequential optimize loop used by the elbow-method config.

use std::cmp::Ordering;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use tracing::{debug, info, instrument};

use narcissus_dtw::{
    BandConstraint, DbaConfig, DistanceMatrix, Dtw, SeriesEnvelope, TimeSeries, TimeSeriesView,
};

use crate::config::{InitStrategy, KMeansConfig, OptimizeConfig};
use crate::elkan::ElkanBounds;
use crate::error::ClusterError;
use crate::inertia::Inertia;
use crate::init::{kmeans_parallel_init, kmeans_plus_plus, kmeans_plus_plus_cached};
use crate::label::ClusterLabel;
use crate::result::{KMeansResult, KResult, OptimizeResult};

// ── Internal run result ───────────────────────────────────────────────────────

/// Result of a single K-means restart.
struct SingleRun {
    assignments: Vec<ClusterLabel>,
    centroids: Vec<TimeSeries>,
    inertia: Inertia,
    converged: bool,
    iterations: usize,
}

// ── assign ────────────────────────────────────────────────────────────────────

/// Assign each series to its nearest centroid and compute total inertia.
///
/// For each series the DTW distance to every centroid is computed; the series
/// is assigned to the centroid with minimum distance. Inertia is the sum of
/// squared distances from each series to its assigned centroid.
///
/// The per-series computation is parallelized with rayon.
///
/// Retained as a reference implementation for tests; production code uses
/// [`assign_pruned`] which applies LB_Keogh / LB_Improved cascading.
#[cfg(test)]
#[instrument(skip(series, centroids, dtw), fields(n = series.len(), k = centroids.len()))]
pub(crate) fn assign(
    series: &[TimeSeries],
    centroids: &[TimeSeries],
    dtw: &Dtw,
) -> (Vec<ClusterLabel>, Inertia) {
    let results: Vec<(ClusterLabel, f64)> = series
        .par_iter()
        .map(|s| {
            let view = s.as_view();
            let mut best_label = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c_idx, centroid) in centroids.iter().enumerate() {
                let d = dtw.distance(view, centroid.as_view()).value();
                if d < best_dist {
                    best_dist = d;
                    best_label = c_idx;
                }
            }
            (ClusterLabel::new(best_label), best_dist)
        })
        .collect();

    let inertia_value: f64 = results.iter().map(|(_, d)| d.powi(2)).sum();
    let assignments: Vec<ClusterLabel> = results.into_iter().map(|(label, _)| label).collect();

    debug!(inertia = inertia_value, "assignment step complete");
    (assignments, Inertia::new(inertia_value))
}

// ── assign_pruned ────────────────────────────────────────────────────────────

/// Assign each series to its nearest centroid using pruned DTW with LB_Keogh
/// and LB_Improved cascading.
///
/// Functionally equivalent to [`assign`], but uses precomputed envelopes and
/// `distance_pruned` to skip centroid candidates whose lower bound exceeds
/// the current best distance. For well-separated clusters this substantially
/// reduces the number of full DTW computations.
#[instrument(skip(series, centroids, dtw, series_envelopes, centroid_envelopes),
             fields(n = series.len(), k = centroids.len()))]
pub(crate) fn assign_pruned(
    series: &[TimeSeries],
    centroids: &[TimeSeries],
    dtw: &Dtw,
    series_envelopes: &[SeriesEnvelope],
    centroid_envelopes: &[SeriesEnvelope],
) -> (Vec<ClusterLabel>, Inertia) {
    let results: Vec<(ClusterLabel, f64)> = series
        .par_iter()
        .zip(series_envelopes.par_iter())
        .map(|(s, s_env)| {
            let view = s.as_view();
            let mut best_label = 0usize;
            let mut best_dist = f64::INFINITY;

            for (c_idx, (centroid, c_env)) in
                centroids.iter().zip(centroid_envelopes.iter()).enumerate()
            {
                let d = dtw
                    .distance_pruned(view, centroid.as_view(), s_env, c_env, best_dist)
                    .value();
                if d < best_dist {
                    best_dist = d;
                    best_label = c_idx;
                }
            }

            (ClusterLabel::new(best_label), best_dist)
        })
        .collect();

    let inertia_value: f64 = results.iter().map(|(_, d)| d.powi(2)).sum();
    let assignments: Vec<ClusterLabel> = results.into_iter().map(|(label, _)| label).collect();

    debug!(inertia = inertia_value, "pruned assignment step complete");
    (assignments, Inertia::new(inertia_value))
}

// ── update ────────────────────────────────────────────────────────────────────

/// Recompute centroids via DBA, rescuing any empty cluster by stealing the
/// series farthest from its current centroid in the largest non-empty cluster.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`ClusterError::Dba`] | DBA centroid computation fails |
#[instrument(skip(series, assignments, dba_config, prev_centroids, dtw),
             fields(k, iteration))]
pub(crate) fn update(
    series: &[TimeSeries],
    assignments: &[ClusterLabel],
    k: usize,
    dba_config: &DbaConfig,
    prev_centroids: &[TimeSeries],
    dtw: &Dtw,
    iteration: usize,
) -> Result<Vec<TimeSeries>, ClusterError> {
    // Step 1: build per-cluster index groups.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, label) in assignments.iter().enumerate() {
        groups[label.index()].push(i);
    }

    // Step 2: rescue empty clusters sequentially.
    // We may need multiple rounds if more than one cluster is empty, so loop
    // until all clusters are non-empty.
    loop {
        let empty_labels: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_empty())
            .map(|(c, _)| c)
            .collect();

        if empty_labels.is_empty() {
            break;
        }

        // Rescue each empty cluster one at a time.
        for empty_label in empty_labels {
            // Find the largest cluster (most members).
            let largest_label = groups
                .iter()
                .enumerate()
                .max_by_key(|(_, g)| g.len())
                .map(|(c, _)| c)
                .expect("k >= 1 guarantees at least one non-empty group");

            if groups[largest_label].len() <= 1 {
                // Cannot steal from a singleton — return an error.
                return Err(ClusterError::EmptyCluster {
                    label: empty_label,
                    iteration,
                });
            }

            // Find the member of the largest cluster farthest from its centroid.
            let centroid_view = prev_centroids[largest_label].as_view();
            let farthest_pos = groups[largest_label]
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| {
                    let da = dtw.distance(series[*a].as_view(), centroid_view).value();
                    let db = dtw.distance(series[*b].as_view(), centroid_view).value();
                    da.total_cmp(&db)
                })
                .map(|(pos, _)| pos)
                .expect("largest group is non-empty");

            // Move the farthest series to the empty cluster.
            let stolen_idx = groups[largest_label].swap_remove(farthest_pos);
            groups[empty_label].push(stolen_idx);

            debug!(
                empty_cluster = empty_label,
                donor_cluster = largest_label,
                stolen_series = stolen_idx,
                "rescued empty cluster"
            );
        }
    }

    // Step 3: run DBA in parallel for each (now non-empty) cluster.
    let new_centroids: Vec<TimeSeries> = groups
        .into_par_iter()
        .map(|group| {
            let views: Vec<TimeSeriesView<'_>> =
                group.iter().map(|&i| series[i].as_view()).collect();
            dba_config
                .average(&views)
                .map(|r| r.centroid)
                .map_err(ClusterError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(new_centroids)
}

// ── run_once ──────────────────────────────────────────────────────────────────

/// Run a single K-means restart seeded with `seed`.
///
/// When `matrix` is `Some`, uses the precomputed pairwise distance matrix for
/// K-means++ initialization instead of computing DTW distances on the fly.
///
/// # Errors
///
/// Propagates [`ClusterError`] from the assign/update loop.
#[instrument(skip(series, config, matrix), fields(k = config.k, seed))]
fn run_once(
    series: &[TimeSeries],
    config: &KMeansConfig,
    seed: u64,
    matrix: Option<&DistanceMatrix>,
) -> Result<SingleRun, ClusterError> {
    let dtw = match config.constraint {
        BandConstraint::Unconstrained => Dtw::unconstrained(),
        BandConstraint::SakoeChibaRadius(r) => Dtw::with_sakoe_chiba(r),
    };

    let dba_config = DbaConfig::new(config.constraint).with_max_iter(config.dba_max_iter);

    // Centroid initialization.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let init_indices = match config.init_strategy {
        InitStrategy::KMeansPlusPlus => match matrix {
            Some(m) => kmeans_plus_plus_cached(series.len(), config.k, m, &mut rng),
            None => kmeans_plus_plus(series, config.k, &dtw, &mut rng),
        },
        InitStrategy::KMeansParallel { oversample_factor } => {
            kmeans_parallel_init(series, config.k, oversample_factor, &dtw, &mut rng)
        }
    };
    let mut centroids: Vec<TimeSeries> =
        init_indices.iter().map(|&i| series[i].clone()).collect();

    let mut assignments: Vec<ClusterLabel> = Vec::new();
    let mut inertia = Inertia::new(f64::INFINITY);
    let mut prev_inertia: Option<f64> = None;
    let mut converged = false;
    let mut iterations = 0usize;

    if config.use_elkan {
        // Elkan path: maintain triangle inequality bounds across iterations.
        let n = series.len();
        let mut bounds = ElkanBounds::new(n, config.k);
        let mut elkan_assignments = vec![0usize; n];

        for iteration in 0..config.max_iter {
            iterations = iteration + 1;

            let old_centroids = centroids.clone();

            let (new_labels, new_inertia, _skipped) =
                bounds.assign(series, &centroids, &dtw, &elkan_assignments);
            assignments = new_labels;
            elkan_assignments = assignments.iter().map(|l| l.index()).collect();

            // Convergence check (skip on the very first iteration).
            if let Some(prev) = prev_inertia
                && (prev - new_inertia.value()).abs() < config.tol
            {
                inertia = new_inertia;
                converged = true;
                debug!(iteration, "converged (elkan)");
                break;
            }

            prev_inertia = Some(new_inertia.value());
            inertia = new_inertia;

            centroids = update(
                series,
                &assignments,
                config.k,
                &dba_config,
                &centroids,
                &dtw,
                iteration,
            )?;

            // Compute centroid shifts for bound updates.
            let centroid_shifts: Vec<f64> = old_centroids
                .iter()
                .zip(centroids.iter())
                .map(|(old, new)| dtw.distance(old.as_view(), new.as_view()).value())
                .collect();

            bounds.update_bounds(&centroid_shifts, &elkan_assignments);

            debug!(
                iteration,
                inertia = inertia.value(),
                "elkan iteration complete"
            );
        }
    } else {
        // Standard path: LB_Keogh/LB_Improved pruned assignment.
        let series_envelopes: Vec<SeriesEnvelope> = series
            .par_iter()
            .map(|s| SeriesEnvelope::compute(s.as_view(), config.constraint))
            .collect();

        for iteration in 0..config.max_iter {
            iterations = iteration + 1;

            // Compute centroid envelopes for the current centroids.
            let centroid_envelopes: Vec<SeriesEnvelope> = centroids
                .iter()
                .map(|c| SeriesEnvelope::compute(c.as_view(), config.constraint))
                .collect();

            let (new_assignments, new_inertia) =
                assign_pruned(series, &centroids, &dtw, &series_envelopes, &centroid_envelopes);
            assignments = new_assignments;

            // Convergence check (skip on the very first iteration).
            if let Some(prev) = prev_inertia
                && (prev - new_inertia.value()).abs() < config.tol
            {
                inertia = new_inertia;
                converged = true;
                debug!(iteration, "converged");
                break;
            }

            prev_inertia = Some(new_inertia.value());
            inertia = new_inertia;

            centroids = update(
                series,
                &assignments,
                config.k,
                &dba_config,
                &centroids,
                &dtw,
                iteration,
            )?;

            debug!(
                iteration,
                inertia = inertia.value(),
                "iteration complete"
            );
        }
    }

    info!(
        seed,
        iterations,
        inertia = inertia.value(),
        converged,
        "single restart complete"
    );

    Ok(SingleRun {
        assignments,
        centroids,
        inertia,
        converged,
        iterations,
    })
}

// ── multi_restart ─────────────────────────────────────────────────────────────

/// Run `config.n_init` independent K-means restarts and return the best result.
///
/// Restarts are executed in parallel. Sub-seeds are derived deterministically
/// from `config.seed` so the overall computation is reproducible.
///
/// When `matrix` is `Some`, the precomputed pairwise distance matrix is passed
/// to each restart for use in K-means++ initialization.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`ClusterError::EmptyCluster`] | A cluster becomes empty and cannot be rescued |
/// | [`ClusterError::Dba`] | A DBA centroid computation fails |
#[instrument(skip(series, config, matrix), fields(k = config.k, n_init = config.n_init))]
pub(crate) fn multi_restart(
    series: &[TimeSeries],
    config: &KMeansConfig,
    matrix: Option<&DistanceMatrix>,
) -> Result<KMeansResult, ClusterError> {
    // Derive per-restart seeds deterministically from the master seed.
    let mut master_rng = ChaCha8Rng::seed_from_u64(config.seed);
    let seeds: Vec<u64> = (0..config.n_init).map(|_| master_rng.r#gen()).collect();

    let n_init = seeds.len();

    // Run all restarts in parallel, collecting results (including errors).
    let results: Vec<Result<SingleRun, ClusterError>> = seeds
        .into_par_iter()
        .map(|seed| run_once(series, config, seed, matrix))
        .collect();

    // Propagate any error encountered, pick the best successful run.
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
        "multi-restart complete"
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

// ── optimize ──────────────────────────────────────────────────────────────────

/// Run K-means for each k in `[config.min_k, config.max_k]` and collect
/// the inertia curve for elbow-method analysis.
///
/// When `config.precompute_matrix` is true, a pairwise DTW distance matrix is
/// computed once and reused for K-means++ initialization across all k values and
/// restarts.
///
/// # Errors
///
/// Propagates the first [`ClusterError`] from any k value.
#[instrument(skip(series, config), fields(min_k = config.min_k, max_k = config.max_k))]
pub(crate) fn optimize(
    series: &[TimeSeries],
    config: &OptimizeConfig,
) -> Result<OptimizeResult, ClusterError> {
    // Optionally precompute the pairwise distance matrix for K-means++ init.
    let matrix = if config.precompute_matrix {
        let dtw = match config.constraint {
            BandConstraint::Unconstrained => Dtw::unconstrained(),
            BandConstraint::SakoeChibaRadius(r) => Dtw::with_sakoe_chiba(r),
        };
        debug!("precomputing pairwise distance matrix");
        Some(dtw.pairwise(series))
    } else {
        None
    };

    let mut results = Vec::with_capacity(config.max_k - config.min_k + 1);

    for k in config.min_k..=config.max_k {
        let k_config = KMeansConfig {
            k,
            constraint: config.constraint,
            n_init: config.n_init,
            max_iter: config.max_iter,
            tol: config.tol,
            seed: config.seed,
            dba_max_iter: config.dba_max_iter,
            use_elkan: false,
            init_strategy: InitStrategy::KMeansPlusPlus,
        };

        let km_result = multi_restart(series, &k_config, matrix.as_ref())?;

        debug!(k, inertia = km_result.inertia.value(), "k complete");
        results.push(KResult { k, inertia: km_result.inertia });
    }

    info!(
        min_k = config.min_k,
        max_k = config.max_k,
        "optimize complete"
    );

    Ok(OptimizeResult { results })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use narcissus_dtw::{BandConstraint, Dtw, SeriesEnvelope, TimeSeries};

    use super::{assign, assign_pruned, multi_restart};
    use crate::config::{InitStrategy, KMeansConfig};

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

    fn config(k: usize, n_init: usize, seed: u64) -> KMeansConfig {
        KMeansConfig::new(k, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(n_init)
            .with_seed(seed)
    }

    #[test]
    fn trivial_k_one() {
        let series = archetype_a();
        let cfg = config(1, 1, 0);
        let result = multi_restart(&series, &cfg, None).unwrap();

        assert_eq!(result.centroids.len(), 1, "should have exactly 1 centroid");
        assert!(
            result.assignments.iter().all(|l| l.index() == 0),
            "all assignments must be cluster 0"
        );
        assert!(result.inertia.value() > 0.0, "inertia must be positive");
    }

    #[test]
    fn three_well_separated_clusters() {
        let series = archetype_a();
        let cfg = config(3, 5, 42);
        let result = multi_restart(&series, &cfg, None).unwrap();

        assert_eq!(result.centroids.len(), 3, "should have 3 centroids");
        assert_eq!(result.assignments.len(), 9, "should have 9 assignments");

        // Build group index → archetype group (0, 1, 2)
        let archetype_group = |idx: usize| -> usize {
            match idx {
                0..=2 => 0,
                3..=5 => 1,
                6..=8 => 2,
                _ => panic!("unexpected index {idx}"),
            }
        };

        // Each cluster label should map to exactly one archetype group.
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

        // Each archetype group must appear exactly once in the mapping.
        let mut seen: Vec<usize> = label_to_archetype
            .into_iter()
            .map(|o| o.expect("every cluster must be occupied"))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2], "each archetype group must form its own cluster");
    }

    #[test]
    fn identical_series_zero_inertia() {
        let series = vec![
            TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap(),
            TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap(),
            TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap(),
        ];
        let cfg = config(1, 1, 7);
        let result = multi_restart(&series, &cfg, None).unwrap();

        assert!(
            result.inertia.value() < 1e-10,
            "inertia for identical series must be near 0, got {}",
            result.inertia.value()
        );
    }

    #[test]
    fn inertia_non_negative() {
        let series = archetype_a();
        let cfg = config(2, 3, 1);
        let result = multi_restart(&series, &cfg, None).unwrap();

        assert!(
            result.inertia.value() >= 0.0,
            "inertia must be non-negative"
        );
    }

    #[test]
    fn deterministic_results() {
        let series = archetype_a();
        let cfg = config(3, 5, 99);

        let r1 = multi_restart(&series, &cfg, None).unwrap();
        let r2 = multi_restart(&series, &cfg, None).unwrap();

        let labels1: Vec<usize> = r1.assignments.iter().map(|l| l.index()).collect();
        let labels2: Vec<usize> = r2.assignments.iter().map(|l| l.index()).collect();

        // Assignments must be identical across identical runs.
        assert_eq!(labels1, labels2, "results must be deterministic");
        assert_eq!(
            r1.inertia.value(),
            r2.inertia.value(),
            "inertia must be deterministic"
        );
    }

    #[test]
    fn convergence_flag_on_easy_data() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(42)
            .with_max_iter(100)
            .with_tol(1e-4);

        let result = multi_restart(&series, &cfg, None).unwrap();
        assert!(result.converged, "should converge on well-separated data");
    }

    #[test]
    fn max_iter_respected() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(1)
            .with_seed(0)
            .with_max_iter(1);

        let result = multi_restart(&series, &cfg, None).unwrap();
        assert!(
            result.iterations <= 1,
            "iterations {} must not exceed max_iter=1",
            result.iterations
        );
    }

    #[test]
    fn label_and_centroid_count_invariants() {
        let series = archetype_a();
        let cfg = config(3, 3, 5);
        let result = multi_restart(&series, &cfg, None).unwrap();

        assert_eq!(result.assignments.len(), 9, "must have one label per series");
        assert_eq!(result.centroids.len(), 3, "must have k centroids");
        assert!(
            result.assignments.iter().all(|l| l.index() < 3),
            "all labels must be in [0, k)"
        );
    }

    #[test]
    fn assign_pruned_matches_assign() {
        let series = archetype_a();
        let constraint = BandConstraint::SakoeChibaRadius(2);
        let dtw = Dtw::with_sakoe_chiba(2);

        // Use three centroids from different groups.
        let centroids = vec![
            series[0].clone(),
            series[3].clone(),
            series[6].clone(),
        ];

        let series_envelopes: Vec<SeriesEnvelope> = series
            .iter()
            .map(|s| SeriesEnvelope::compute(s.as_view(), constraint))
            .collect();
        let centroid_envelopes: Vec<SeriesEnvelope> = centroids
            .iter()
            .map(|c| SeriesEnvelope::compute(c.as_view(), constraint))
            .collect();

        let (labels_plain, inertia_plain) = assign(&series, &centroids, &dtw);
        let (labels_pruned, inertia_pruned) =
            assign_pruned(&series, &centroids, &dtw, &series_envelopes, &centroid_envelopes);

        let plain_indices: Vec<usize> = labels_plain.iter().map(|l| l.index()).collect();
        let pruned_indices: Vec<usize> = labels_pruned.iter().map(|l| l.index()).collect();

        assert_eq!(
            plain_indices, pruned_indices,
            "pruned assign must produce identical assignments as plain assign"
        );
        assert!(
            (inertia_plain.value() - inertia_pruned.value()).abs() < 1e-10,
            "inertia mismatch: plain={} pruned={}",
            inertia_plain.value(),
            inertia_pruned.value()
        );
    }

    #[test]
    fn pruned_dtw_deterministic() {
        let series = archetype_a();
        let cfg = config(3, 1, 77);

        let r1 = multi_restart(&series, &cfg, None).unwrap();
        let r2 = multi_restart(&series, &cfg, None).unwrap();

        let labels1: Vec<usize> = r1.assignments.iter().map(|l| l.index()).collect();
        let labels2: Vec<usize> = r2.assignments.iter().map(|l| l.index()).collect();

        assert_eq!(labels1, labels2, "pruned DTW results must be deterministic");
        assert_eq!(
            r1.inertia.value(),
            r2.inertia.value(),
            "pruned DTW inertia must be deterministic"
        );
    }

    #[test]
    fn parallel_init_three_clusters() {
        // Run multi_restart with KMeansParallel on archetype_a (k=3) and
        // verify that the clustering produces a correct 3-3-3 partition.
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(42)
            .with_init_strategy(InitStrategy::KMeansParallel { oversample_factor: 2.0 });

        let result = multi_restart(&series, &cfg, None).unwrap();

        assert_eq!(result.centroids.len(), 3, "should have 3 centroids");
        assert_eq!(result.assignments.len(), 9, "should have 9 assignments");

        // Archetype group for each series index.
        let archetype_group = |idx: usize| -> usize {
            match idx {
                0..=2 => 0,
                3..=5 => 1,
                6..=8 => 2,
                _ => panic!("unexpected index {idx}"),
            }
        };

        // Each cluster label should map to exactly one archetype group.
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

        // Each archetype group must appear exactly once in the mapping.
        let mut seen: Vec<usize> = label_to_archetype
            .into_iter()
            .map(|o| o.expect("every cluster must be occupied"))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2], "each archetype group must form its own cluster");
    }
}
