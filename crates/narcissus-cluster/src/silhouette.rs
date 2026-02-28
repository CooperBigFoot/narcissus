//! Silhouette score for evaluating clustering quality.
//!
//! The silhouette score measures how similar a series is to its own cluster
//! compared to other clusters. Values range from -1 (poor) to +1 (perfect).

use rayon::prelude::*;

use narcissus_dtw::{Dtw, TimeSeries};

use crate::error::ClusterError;
use crate::label::ClusterLabel;

/// Silhouette score for a single sample.
#[derive(Debug, Clone)]
pub struct SampleSilhouette {
    /// Mean intra-cluster distance (a(i)).
    pub a: f64,
    /// Mean nearest-cluster distance (b(i)).
    pub b: f64,
    /// Silhouette coefficient: (b - a) / max(a, b). In [-1, 1].
    pub score: f64,
    /// Cluster assignment of this sample.
    pub cluster: ClusterLabel,
}

/// Result of silhouette score computation.
#[derive(Debug, Clone)]
pub struct SilhouetteScore {
    /// Per-sample silhouette scores.
    pub per_sample: Vec<SampleSilhouette>,
    /// Mean silhouette score across all samples.
    pub mean_score: f64,
    /// Mean silhouette score per cluster.
    pub per_cluster: Vec<f64>,
}

/// Compute silhouette scores for a clustering result.
///
/// For each sample `i`:
/// - `a(i)` = mean DTW distance from `i` to all other members of its cluster
/// - `b(i)` = min over all other clusters c of (mean DTW distance from `i` to members of c)
/// - `s(i)` = `(b(i) - a(i)) / max(a(i), b(i))`
///
/// Per-sample computation is parallelized with rayon.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`ClusterError::SingleCluster`] | Fewer than 2 distinct clusters |
#[must_use = "compute_silhouette returns a Result that should be used"]
pub fn compute_silhouette(
    series: &[TimeSeries],
    assignments: &[ClusterLabel],
    k: usize,
    dtw: &Dtw,
) -> Result<SilhouetteScore, ClusterError> {
    // Verify we have at least 2 clusters with members
    let mut cluster_sizes = vec![0usize; k];
    for label in assignments {
        cluster_sizes[label.index()] += 1;
    }
    let n_nonempty = cluster_sizes.iter().filter(|&&s| s > 0).count();
    if n_nonempty < 2 {
        return Err(ClusterError::SingleCluster { n_clusters: n_nonempty });
    }

    let n = series.len();

    // Build per-cluster member indices
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, label) in assignments.iter().enumerate() {
        members[label.index()].push(i);
    }

    // Compute per-sample silhouette in parallel
    let per_sample: Vec<SampleSilhouette> = (0..n)
        .into_par_iter()
        .map(|i| {
            let my_cluster = assignments[i].index();
            let view_i = series[i].as_view();

            // a(i): mean distance to same-cluster members (excluding self)
            let a = if members[my_cluster].len() <= 1 {
                0.0
            } else {
                let sum: f64 = members[my_cluster]
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| dtw.distance(view_i, series[j].as_view()).value())
                    .sum();
                sum / (members[my_cluster].len() - 1) as f64
            };

            // b(i): min mean distance to any other cluster
            let b = (0..k)
                .filter(|&c| c != my_cluster && !members[c].is_empty())
                .map(|c| {
                    let sum: f64 = members[c]
                        .iter()
                        .map(|&j| dtw.distance(view_i, series[j].as_view()).value())
                        .sum();
                    sum / members[c].len() as f64
                })
                .fold(f64::INFINITY, f64::min);

            // s(i) = (b - a) / max(a, b)
            let score = if a.max(b) == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            };

            SampleSilhouette {
                a,
                b,
                score,
                cluster: assignments[i],
            }
        })
        .collect();

    // Mean score
    let mean_score = per_sample.iter().map(|s| s.score).sum::<f64>() / n as f64;

    // Per-cluster mean
    let per_cluster: Vec<f64> = (0..k)
        .map(|c| {
            if members[c].is_empty() {
                0.0
            } else {
                let sum: f64 = members[c].iter().map(|&i| per_sample[i].score).sum();
                sum / members[c].len() as f64
            }
        })
        .collect();

    Ok(SilhouetteScore {
        per_sample,
        mean_score,
        per_cluster,
    })
}

#[cfg(test)]
mod tests {
    use narcissus_dtw::{Dtw, TimeSeries};

    use crate::error::ClusterError;
    use crate::label::ClusterLabel;

    use super::compute_silhouette;

    /// Build 9 series in 3 tight archetype clusters: around 0, 5, and 10.
    fn archetype_series() -> Vec<TimeSeries> {
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

    fn perfect_assignments() -> Vec<ClusterLabel> {
        vec![
            ClusterLabel::new(0),
            ClusterLabel::new(0),
            ClusterLabel::new(0),
            ClusterLabel::new(1),
            ClusterLabel::new(1),
            ClusterLabel::new(1),
            ClusterLabel::new(2),
            ClusterLabel::new(2),
            ClusterLabel::new(2),
        ]
    }

    fn scrambled_assignments() -> Vec<ClusterLabel> {
        // Interleaved — each series is assigned to a different cluster from its archetype
        vec![
            ClusterLabel::new(0),
            ClusterLabel::new(1),
            ClusterLabel::new(2),
            ClusterLabel::new(0),
            ClusterLabel::new(1),
            ClusterLabel::new(2),
            ClusterLabel::new(0),
            ClusterLabel::new(1),
            ClusterLabel::new(2),
        ]
    }

    #[test]
    fn perfect_clusters_near_one() {
        let series = archetype_series();
        let assignments = perfect_assignments();
        let dtw = Dtw::unconstrained();

        let result = compute_silhouette(&series, &assignments, 3, &dtw).unwrap();

        assert!(
            result.mean_score > 0.8,
            "expected mean silhouette > 0.8 for perfect clusters, got {}",
            result.mean_score
        );
    }

    #[test]
    fn random_assignments_low() {
        let series = archetype_series();
        let perfect_assignments = perfect_assignments();
        let scrambled = scrambled_assignments();
        let dtw = Dtw::unconstrained();

        let perfect = compute_silhouette(&series, &perfect_assignments, 3, &dtw).unwrap();
        let scrambled = compute_silhouette(&series, &scrambled, 3, &dtw).unwrap();

        assert!(
            scrambled.mean_score < 0.5,
            "expected scrambled mean silhouette < 0.5, got {}",
            scrambled.mean_score
        );
        assert!(
            perfect.mean_score > scrambled.mean_score,
            "perfect clustering ({}) should score higher than scrambled ({})",
            perfect.mean_score,
            scrambled.mean_score
        );
    }

    #[test]
    fn single_cluster_error() {
        let series = archetype_series();
        // All series assigned to cluster 0, k=1
        let assignments: Vec<ClusterLabel> =
            (0..9).map(|_| ClusterLabel::new(0)).collect();
        let dtw = Dtw::unconstrained();

        let result = compute_silhouette(&series, &assignments, 1, &dtw);

        assert!(
            matches!(result, Err(ClusterError::SingleCluster { n_clusters: 1 })),
            "expected SingleCluster error, got {:?}",
            result
        );
    }

    #[test]
    fn per_sample_scores_in_range() {
        let series = archetype_series();
        let assignments = perfect_assignments();
        let dtw = Dtw::unconstrained();

        let result = compute_silhouette(&series, &assignments, 3, &dtw).unwrap();

        for (i, sample) in result.per_sample.iter().enumerate() {
            assert!(
                sample.score >= -1.0 && sample.score <= 1.0,
                "sample {i} score {} is out of [-1, 1]",
                sample.score
            );
        }
    }

    #[test]
    fn per_cluster_count_matches_k() {
        let series = archetype_series();
        let assignments = perfect_assignments();
        let dtw = Dtw::unconstrained();
        let k = 3;

        let result = compute_silhouette(&series, &assignments, k, &dtw).unwrap();

        assert_eq!(
            result.per_cluster.len(),
            k,
            "per_cluster length {} != k {}",
            result.per_cluster.len(),
            k
        );
    }

    #[test]
    fn per_cluster_count_matches_k_with_band() {
        let series = archetype_series();
        let assignments = perfect_assignments();
        let dtw = Dtw::with_sakoe_chiba(2);
        let k = 3;

        let result = compute_silhouette(&series, &assignments, k, &dtw).unwrap();

        assert_eq!(result.per_cluster.len(), k);
        assert!(result.mean_score > 0.5);
    }
}
