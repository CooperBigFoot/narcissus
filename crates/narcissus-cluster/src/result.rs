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
    /// Results for each k value tested.
    pub results: Vec<KResult>,
}
