//! K-means clustering with DTW distance metric.
//!
//! Provides DTW-based K-means clustering with k-means++ initialization,
//! multi-restart optimization, and elbow-method cluster count selection.

mod config;
mod elkan;
mod error;
mod inertia;
mod init;
mod kmeans;
mod label;
mod minibatch;
mod result;
mod silhouette;

pub use config::{InitStrategy, KMeansConfig, MiniBatchConfig, OptimizeConfig};
pub use error::ClusterError;
pub use inertia::Inertia;
pub use label::ClusterLabel;
pub use result::{KMeansResult, KResult, OptimizeResult};
pub use silhouette::{compute_silhouette, SampleSilhouette, SilhouetteScore};

#[cfg(test)]
mod tests {
    use narcissus_dtw::{BandConstraint, Dtw, TimeSeries};

    use crate::{ClusterLabel, KMeansConfig, OptimizeConfig};

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

    // Test 1: pipeline consistency — recompute inertia manually from result
    #[test]
    fn inertia_matches_manual_computation() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        // Recompute inertia manually: sum of squared DTW distances to assigned centroids
        let dtw = Dtw::unconstrained();
        let manual_inertia: f64 = series
            .iter()
            .zip(&result.assignments)
            .map(|(s, label)| {
                let d = dtw.distance(s.as_view(), result.centroids[label.index()].as_view()).value();
                d.powi(2)
            })
            .sum();

        assert!(
            (result.inertia.value() - manual_inertia).abs() < 1e-6,
            "reported inertia {} differs from manual computation {}",
            result.inertia.value(),
            manual_inertia
        );
    }

    // Test 2: centroids are valid TimeSeries (correct length, finite)
    #[test]
    fn centroids_are_valid_time_series() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(3)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        for (i, centroid) in result.centroids.iter().enumerate() {
            assert_eq!(
                centroid.len(),
                4,
                "centroid {i} length {} != expected 4",
                centroid.len()
            );
            // Every value should be finite (TimeSeries guarantees this, but verify)
            for val in centroid.as_ref() {
                assert!(val.is_finite(), "centroid {i} contains non-finite value");
            }
        }
    }

    // Test 3: labels index valid centroids
    #[test]
    fn labels_index_valid_centroids() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(3)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        let k = result.centroids.len();
        for (i, label) in result.assignments.iter().enumerate() {
            assert!(
                label.index() < k,
                "series {i} assigned to label {} but only {k} centroids exist",
                label.index()
            );
        }
    }

    // Test 4: elbow invariant — inertia non-increasing with k
    #[test]
    fn elbow_inertia_non_increasing() {
        let series = archetype_a();
        let cfg = OptimizeConfig::new(1, 5, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(3)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        assert_eq!(result.results.len(), 5, "should have results for k=1..=5");

        // Inertia should generally be non-increasing with k.
        // Due to randomness, allow small violations but check the overall trend.
        for window in result.results.windows(2) {
            let prev = window[0].inertia.value();
            let next = window[1].inertia.value();
            // Allow small relative increase (< 5%) due to stochastic initialization
            assert!(
                next <= prev * 1.05,
                "inertia for k={} ({}) significantly exceeds k={} ({})",
                window[1].k, next, window[0].k, prev
            );
        }
    }

    // Test 5: k=n gives inertia near 0
    #[test]
    fn k_equals_n_near_zero_inertia() {
        let series = archetype_a();
        let n = series.len();
        let cfg = KMeansConfig::new(n, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(1)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        assert!(
            result.inertia.value() < 1e-6,
            "k=n should give near-zero inertia, got {}",
            result.inertia.value()
        );
    }

    // Test 6: correct record count in optimize
    #[test]
    fn optimize_result_record_count() {
        let series = archetype_a();
        let cfg = OptimizeConfig::new(2, 7, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(2)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        assert_eq!(
            result.results.len(),
            6,
            "should have 6 results for k=2..=7"
        );
        // Check k values are sequential
        for (i, kr) in result.results.iter().enumerate() {
            assert_eq!(kr.k, i + 2, "k values should be sequential starting at 2");
        }
    }

    // Test 7: fit() returns TooFewSeries when n < k
    #[test]
    fn fit_too_few_series() {
        let series = vec![
            TimeSeries::new(vec![1.0, 2.0]).unwrap(),
            TimeSeries::new(vec![3.0, 4.0]).unwrap(),
        ];
        let cfg = KMeansConfig::new(5, BandConstraint::Unconstrained).unwrap();
        let result = cfg.fit(&series);
        assert!(matches!(
            result,
            Err(crate::ClusterError::TooFewSeries { n_series: 2, k: 5 })
        ));
    }

    // Test 8: cluster_sizes sums to n
    #[test]
    fn cluster_sizes_sum_to_n() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(3)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        let total: usize = result.cluster_sizes().iter().sum();
        assert_eq!(total, series.len(), "cluster sizes must sum to n");
    }

    // Test 9: members returns consistent indices
    #[test]
    fn members_are_consistent() {
        let series = archetype_a();
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(3)
            .with_seed(42);
        let result = cfg.fit(&series).unwrap();

        let mut all_members = Vec::new();
        for c in 0..3 {
            let members = result.members(ClusterLabel::new(c));
            for &idx in &members {
                assert_eq!(
                    result.assignments[idx].index(), c,
                    "series {idx} reported as member of cluster {c} but assigned to {}",
                    result.assignments[idx].index()
                );
            }
            all_members.extend(members);
        }
        all_members.sort();
        assert_eq!(all_members, (0..9).collect::<Vec<_>>(), "all series must appear exactly once");
    }
}
