//! Error types for DTW computation and DBA averaging.

/// Errors from DTW distance computation and time series validation.
#[derive(Debug, thiserror::Error)]
pub enum DtwError {
    /// Returned when an empty slice is provided as a time series.
    #[error("time series must be non-empty")]
    EmptySeries,

    /// Returned when a time series contains NaN, infinity, or negative infinity.
    #[error("time series contains non-finite value at index {index}")]
    NonFiniteValue {
        /// Position of the first non-finite value found.
        index: usize,
    },
}

/// Errors from DBA barycenter averaging.
#[derive(Debug, thiserror::Error)]
pub enum DbaError {
    /// Returned when `average()` is called with an empty slice of series.
    #[error("cannot compute barycenter of an empty cluster")]
    EmptyCluster,

    /// Wraps a DTW error encountered during alignment.
    #[error("DTW error during DBA: {0}")]
    Dtw(#[from] DtwError),
}
