//! Result types for K-means clustering and elbow-method optimization.

use narcissus_dtw::TimeSeries;

use crate::inertia::Inertia;
use crate::label::ClusterLabel;

/// Result of a single K-means clustering run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster assignment for each input series.
    pub assignments: Vec<ClusterLabel>,
    /// Centroid time series for each cluster.
    pub centroids: Vec<TimeSeries>,
    /// Total inertia (sum of squared distances to assigned centroids).
    pub inertia: Inertia,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// Number of iterations performed in the best run.
    pub iterations: usize,
    /// Number of random restarts actually executed.
    pub n_init_used: usize,
}

impl KMeansResult {
    /// Return the number of series assigned to each cluster.
    ///
    /// The returned vec has length equal to `centroids.len()`. Entry `i` holds the
    /// count of series assigned to cluster `i`.
    #[must_use]
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let k = self.centroids.len();
        let mut sizes = vec![0usize; k];
        for label in &self.assignments {
            sizes[label.index()] += 1;
        }
        sizes
    }

    /// Return the indices of all series assigned to `label`.
    ///
    /// Indices correspond to positions in the original `series` slice passed to `fit`.
    #[must_use]
    pub fn members(&self, label: ClusterLabel) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| if l == label { Some(i) } else { None })
            .collect()
    }
}

/// Inertia result for a single k value.
#[derive(Debug, Clone)]
pub struct KResult {
    /// The number of clusters.
    pub k: usize,
    /// The best inertia achieved for this k.
    pub inertia: Inertia,
}

/// Result of elbow-method cluster count optimization.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Results for each k value tested, ordered by ascending k.
    pub results: Vec<KResult>,
}

impl OptimizeResult {
    /// Return the best k using the maximum second-derivative (elbow) heuristic.
    ///
    /// For each interior point `i` (not the first or last), the discrete second
    /// derivative is:
    ///
    /// ```text
    /// d2[i] = inertia[i-1] - 2 * inertia[i] + inertia[i+1]
    /// ```
    ///
    /// The k with the highest `d2` value is the elbow — it marks the steepest
    /// drop-off in the rate of inertia improvement.
    ///
    /// - Returns `None` when `results` is empty.
    /// - Returns the first k when `results` has fewer than 3 entries (no interior
    ///   points to compare).
    #[must_use]
    pub fn best_k(&self) -> Option<usize> {
        match self.results.len() {
            0 => None,
            1 | 2 => Some(self.results[0].k),
            n => {
                // Collect inertia values as f64 for arithmetic.
                let inertias: Vec<f64> =
                    self.results.iter().map(|r| r.inertia.value()).collect();

                // Find the interior index with the maximum second derivative.
                let best_idx = (1..n - 1)
                    .max_by(|&i, &j| {
                        let d2_i = inertias[i - 1] - 2.0 * inertias[i] + inertias[i + 1];
                        let d2_j = inertias[j - 1] - 2.0 * inertias[j] + inertias[j + 1];
                        d2_i.total_cmp(&d2_j)
                    })
                    .unwrap(); // safe: range is non-empty when n >= 3

                Some(self.results[best_idx].k)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use narcissus_dtw::TimeSeries;

    use super::{KMeansResult, KResult, OptimizeResult};
    use crate::inertia::Inertia;
    use crate::label::ClusterLabel;

    /// Build a minimal [`KMeansResult`] with the given assignments and k centroids.
    fn make_result(assignments: Vec<usize>, k: usize) -> KMeansResult {
        let centroids: Vec<TimeSeries> = (0..k)
            .map(|i| TimeSeries::new(vec![i as f64 + 1.0]).unwrap())
            .collect();
        KMeansResult {
            assignments: assignments.iter().map(|&i| ClusterLabel::new(i)).collect(),
            centroids,
            inertia: Inertia::new(0.0),
            converged: true,
            iterations: 1,
            n_init_used: 1,
        }
    }

    #[test]
    fn cluster_sizes_basic() {
        // 5 series: cluster 0 gets 3, cluster 1 gets 2.
        let result = make_result(vec![0, 1, 0, 0, 1], 2);
        let sizes = result.cluster_sizes();
        assert_eq!(sizes, vec![3, 2]);
    }

    #[test]
    fn members_basic() {
        // Series 0, 2, 3 → cluster 0; series 1, 4 → cluster 1.
        let result = make_result(vec![0, 1, 0, 0, 1], 2);
        let mut m0 = result.members(ClusterLabel::new(0));
        m0.sort();
        assert_eq!(m0, vec![0, 2, 3]);

        let mut m1 = result.members(ClusterLabel::new(1));
        m1.sort();
        assert_eq!(m1, vec![1, 4]);
    }

    /// Build an [`OptimizeResult`] from (k, inertia) pairs.
    fn make_opt(pairs: &[(usize, f64)]) -> OptimizeResult {
        OptimizeResult {
            results: pairs
                .iter()
                .map(|&(k, v)| KResult { k, inertia: Inertia::new(v) })
                .collect(),
        }
    }

    #[test]
    fn best_k_three_points() {
        // inertias: [100.0, 30.0, 25.0] for k = [1, 2, 3]
        // d2 for k=2 (index 1) = 100 - 2*30 + 25 = 65
        // Only one interior point, so best_k == 2.
        let opt = make_opt(&[(1, 100.0), (2, 30.0), (3, 25.0)]);
        assert_eq!(opt.best_k(), Some(2));
    }

    #[test]
    fn best_k_empty() {
        let opt = make_opt(&[]);
        assert_eq!(opt.best_k(), None);
    }

    #[test]
    fn best_k_single() {
        let opt = make_opt(&[(3, 42.0)]);
        assert_eq!(opt.best_k(), Some(3));
    }

    #[test]
    fn best_k_two_results() {
        let opt = make_opt(&[(2, 80.0), (4, 40.0)]);
        assert_eq!(opt.best_k(), Some(2));
    }
}
