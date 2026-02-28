//! Accuracy regression tests for narcissus-cluster.
//!
//! These tests verify that algorithmic changes do not degrade K-means clustering
//! quality on the archetype 9-series dataset with 3 tight clusters.

use narcissus_cluster::{KMeansConfig, OptimizeConfig};
use narcissus_dtw::{BandConstraint, TimeSeries};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Nine series arranged in 3 tight clusters around 0, 5, and 10.
fn archetype_data() -> Vec<TimeSeries> {
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

// ---------------------------------------------------------------------------
// a) three_clusters_produce_three_three_three_split
// ---------------------------------------------------------------------------

/// K=3 must assign exactly 3 series to each cluster, with each cluster
/// corresponding to exactly one archetype group (indices 0-2, 3-5, 6-8).
#[test]
fn three_clusters_produce_three_three_three_split() {
    let series = archetype_data();
    let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
        .unwrap()
        .with_n_init(10)
        .with_seed(42);
    let result = cfg.fit(&series).unwrap();

    // Must produce exactly 3 distinct labels.
    let unique_labels: std::collections::HashSet<usize> =
        result.assignments.iter().map(|l| l.index()).collect();
    assert_eq!(unique_labels.len(), 3, "expected 3 unique cluster labels");

    // Each cluster must have exactly 3 members.
    let sizes = result.cluster_sizes();
    for (i, &size) in sizes.iter().enumerate() {
        assert_eq!(size, 3, "cluster {i} has {size} members, expected 3");
    }

    // Each of the three archetype groups must map to exactly one cluster label.
    let archetype_groups = [[0usize, 1, 2], [3, 4, 5], [6, 7, 8]];
    for group in &archetype_groups {
        let group_labels: std::collections::HashSet<usize> =
            group.iter().map(|&i| result.assignments[i].index()).collect();
        assert_eq!(
            group_labels.len(),
            1,
            "archetype group {group:?} is split across multiple clusters: {group_labels:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// b) inertia_below_baseline_threshold
// ---------------------------------------------------------------------------

/// K=3 inertia must be below a generous 2x threshold of the observed value.
///
/// Reference inertia ~0.04 was observed from the implementation. Threshold is
/// set at 0.10 to give a generous margin while still catching regressions.
#[test]
fn inertia_below_baseline_threshold() {
    // Reference: observed inertia ≈ 0.04 with seed=42, n_init=10.
    // Threshold at 0.10 is ~2.5x the reference, giving stochastic tolerance.
    let threshold = 0.10_f64;

    let series = archetype_data();
    let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
        .unwrap()
        .with_n_init(10)
        .with_seed(42);
    let result = cfg.fit(&series).unwrap();

    assert!(
        result.inertia.value() < threshold,
        "inertia {} >= threshold {}",
        result.inertia.value(),
        threshold
    );
}

// ---------------------------------------------------------------------------
// c) elbow_best_k_returns_three
// ---------------------------------------------------------------------------

/// Elbow method on k=2..9 must identify k=3 as the best cluster count.
///
/// Note: using k=2..9 (not k=1..9) because the second derivative at k=2
/// is dominated by the large drop from k=1 to k=2 when k=1 is included,
/// while k=2..9 correctly identifies k=3 as the structural elbow.
#[test]
fn elbow_best_k_returns_three() {
    let series = archetype_data();
    let cfg = OptimizeConfig::new(2, 9, BandConstraint::Unconstrained)
        .unwrap()
        .with_n_init(5)
        .with_seed(42);
    let result = cfg.fit(&series).unwrap();

    assert_eq!(
        result.best_k(),
        Some(3),
        "expected best_k=3, got {:?}",
        result.best_k()
    );
}

// ---------------------------------------------------------------------------
// d) inertia_monotonically_non_increasing
// ---------------------------------------------------------------------------

/// Inertia for k=1..5 must be non-increasing (allowing 5% tolerance for stochastic effects).
#[test]
fn inertia_monotonically_non_increasing() {
    let series = archetype_data();
    let cfg = OptimizeConfig::new(1, 5, BandConstraint::Unconstrained)
        .unwrap()
        .with_n_init(5)
        .with_seed(42);
    let result = cfg.fit(&series).unwrap();

    assert_eq!(result.results.len(), 5, "expected 5 results for k=1..=5");

    for window in result.results.windows(2) {
        let prev = window[0].inertia.value();
        let next = window[1].inertia.value();
        assert!(
            next <= prev * 1.05,
            "inertia for k={} ({}) significantly exceeds k={} ({})",
            window[1].k,
            next,
            window[0].k,
            prev
        );
    }
}

// ---------------------------------------------------------------------------
// e) deterministic_across_runs
// ---------------------------------------------------------------------------

/// Same config and seed must produce identical assignments and inertia across runs.
#[test]
fn deterministic_across_runs() {
    let series = archetype_data();
    let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
        .unwrap()
        .with_n_init(5)
        .with_seed(42);

    let result1 = cfg.fit(&series).unwrap();
    let result2 = cfg.fit(&series).unwrap();

    assert_eq!(
        result1.assignments.iter().map(|l| l.index()).collect::<Vec<_>>(),
        result2.assignments.iter().map(|l| l.index()).collect::<Vec<_>>(),
        "assignments differ across runs with same seed"
    );

    assert!(
        (result1.inertia.value() - result2.inertia.value()).abs() < 1e-10,
        "inertia differs across runs: {} vs {}",
        result1.inertia.value(),
        result2.inertia.value()
    );
}
