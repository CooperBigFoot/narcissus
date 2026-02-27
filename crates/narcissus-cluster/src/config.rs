use narcissus_dtw::BandConstraint;

/// Configuration for K-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub(crate) k: usize,
    pub(crate) constraint: BandConstraint,
    pub(crate) n_init: usize,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
    pub(crate) seed: u64,
    pub(crate) dba_max_iter: usize,
}

/// Configuration for elbow-method cluster count optimization.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    pub(crate) min_k: usize,
    pub(crate) max_k: usize,
    pub(crate) constraint: BandConstraint,
    pub(crate) n_init: usize,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
    pub(crate) seed: u64,
    pub(crate) dba_max_iter: usize,
}
