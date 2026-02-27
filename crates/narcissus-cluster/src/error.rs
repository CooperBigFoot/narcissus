use narcissus_dtw::DbaError;

/// Errors from K-means clustering operations.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    /// Returned when k is zero.
    #[error("k must be at least 1, got {k}")]
    InvalidK {
        /// The invalid k value provided.
        k: usize,
    },

    /// Returned when fewer series are provided than the requested k.
    #[error("need at least {k} series to form {k} clusters, got {n_series}")]
    TooFewSeries {
        /// Number of series provided.
        n_series: usize,
        /// Requested number of clusters.
        k: usize,
    },

    /// Returned when min_k exceeds max_k in an optimization range.
    #[error("min_k ({min_k}) must not exceed max_k ({max_k})")]
    InvalidKRange {
        /// The minimum k value.
        min_k: usize,
        /// The maximum k value.
        max_k: usize,
    },

    /// Returned when a cluster becomes empty and cannot be rescued.
    #[error("cluster {label} became empty at iteration {iteration}")]
    EmptyCluster {
        /// The cluster label that became empty.
        label: usize,
        /// The iteration at which the cluster became empty.
        iteration: usize,
    },

    /// Wraps a DBA error encountered during centroid computation.
    #[error("DBA error during centroid update: {0}")]
    Dba(#[from] DbaError),
}
