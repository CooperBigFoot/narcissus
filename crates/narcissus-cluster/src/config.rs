//! Configuration builders for K-means and elbow-method cluster count optimization.

use narcissus_dtw::{BandConstraint, TimeSeries};

use crate::error::ClusterError;
use crate::result::{KMeansResult, OptimizeResult};

/// Initialization strategy for K-means centroid selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitStrategy {
    /// Standard K-means++ initialization (default).
    KMeansPlusPlus,
    /// K-Means‖ (K-Means Parallel) oversampled D²-sampling initialization.
    /// Runs O(log k) rounds, each sampling `oversample_factor * k` candidates.
    KMeansParallel {
        /// Number of candidates per round = oversample_factor * k.
        oversample_factor: f64,
    },
}

/// Configuration for K-means clustering.
///
/// Construct via [`KMeansConfig::new`], then chain `with_*` methods to override defaults.
///
/// # Defaults
///
/// | Parameter       | Default                    |
/// |-----------------|----------------------------|
/// | `n_init`        | 10                         |
/// | `max_iter`      | 75                         |
/// | `tol`           | 1e-4                       |
/// | `seed`          | 42                         |
/// | `dba_max_iter`  | 10                         |
/// | `use_elkan`     | false                      |
/// | `init_strategy` | `InitStrategy::KMeansPlusPlus` |
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub(crate) k: usize,
    pub(crate) constraint: BandConstraint,
    pub(crate) n_init: usize,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
    pub(crate) seed: u64,
    pub(crate) dba_max_iter: usize,
    pub(crate) use_elkan: bool,
    pub(crate) init_strategy: InitStrategy,
}

impl KMeansConfig {
    /// Create a new K-means configuration with the given cluster count and band constraint.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::InvalidK`] | `k` is zero |
    pub fn new(k: usize, constraint: BandConstraint) -> Result<Self, ClusterError> {
        if k == 0 {
            return Err(ClusterError::InvalidK { k });
        }
        Ok(Self {
            k,
            constraint,
            n_init: 10,
            max_iter: 75,
            tol: 1e-4,
            seed: 42,
            dba_max_iter: 10,
            use_elkan: false,
            init_strategy: InitStrategy::KMeansPlusPlus,
        })
    }

    /// Set the number of independent restarts. Higher values reduce the risk of
    /// converging to a poor local minimum.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the maximum number of EM iterations per restart.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance. Iteration stops when inertia improvement
    /// falls below this threshold.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed used for k-means++ initialization and restart shuffling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the maximum number of DBA iterations used when computing centroids.
    #[must_use]
    pub fn with_dba_max_iter(mut self, dba_max_iter: usize) -> Self {
        self.dba_max_iter = dba_max_iter;
        self
    }

    /// Enable or disable Elkan's triangle inequality acceleration for the
    /// assignment step.
    ///
    /// When enabled, per-series distance bounds are maintained across iterations
    /// to skip redundant DTW computations. Most beneficial when `k` and `n` are
    /// large. Disabled by default.
    #[must_use]
    pub fn with_use_elkan(mut self, use_elkan: bool) -> Self {
        self.use_elkan = use_elkan;
        self
    }

    /// Set the centroid initialization strategy.
    ///
    /// Defaults to [`InitStrategy::KMeansPlusPlus`]. Use
    /// [`InitStrategy::KMeansParallel`] for faster initialization on large
    /// datasets by oversampling candidates in O(log k) parallel rounds.
    #[must_use]
    pub fn with_init_strategy(mut self, init_strategy: InitStrategy) -> Self {
        self.init_strategy = init_strategy;
        self
    }

    /// Return the number of clusters.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the band constraint used for DTW distance computation.
    #[must_use]
    pub fn constraint(&self) -> BandConstraint {
        self.constraint
    }

    /// Return the number of independent restarts.
    #[must_use]
    pub fn n_init(&self) -> usize {
        self.n_init
    }

    /// Return the maximum number of EM iterations per restart.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the maximum number of DBA iterations used when computing centroids.
    #[must_use]
    pub fn dba_max_iter(&self) -> usize {
        self.dba_max_iter
    }

    /// Return whether Elkan's triangle inequality acceleration is enabled.
    #[must_use]
    pub fn use_elkan(&self) -> bool {
        self.use_elkan
    }

    /// Return the centroid initialization strategy.
    #[must_use]
    pub fn init_strategy(&self) -> InitStrategy {
        self.init_strategy
    }

    /// Cluster `series` using this configuration.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::TooFewSeries`] | `series.len() < k` |
    /// | [`ClusterError::EmptyCluster`] | A cluster becomes empty and cannot be rescued |
    /// | [`ClusterError::Dba`] | A DBA centroid update fails |
    pub fn fit(&self, series: &[TimeSeries]) -> Result<KMeansResult, ClusterError> {
        let n = series.len();
        if n < self.k {
            return Err(ClusterError::TooFewSeries { n_series: n, k: self.k });
        }
        crate::kmeans::multi_restart(series, self, None)
    }
}

/// Configuration for elbow-method cluster count optimization.
///
/// Runs K-means for each k in `[min_k, max_k]` and selects the k at the
/// elbow of the inertia curve using maximum second-derivative detection.
///
/// # Defaults
///
/// | Parameter      | Default |
/// |----------------|---------|
/// | `n_init`       | 10      |
/// | `max_iter`     | 75      |
/// | `tol`          | 1e-4    |
/// | `seed`         | 42      |
/// | `dba_max_iter`       | 10      |
/// | `precompute_matrix`  | true    |
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
    pub(crate) precompute_matrix: bool,
}

impl OptimizeConfig {
    /// Create a new optimization configuration for the cluster range `[min_k, max_k]`.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::InvalidK`] | `min_k` is zero |
    /// | [`ClusterError::InvalidKRange`] | `min_k > max_k` |
    pub fn new(
        min_k: usize,
        max_k: usize,
        constraint: BandConstraint,
    ) -> Result<Self, ClusterError> {
        if min_k == 0 {
            return Err(ClusterError::InvalidK { k: min_k });
        }
        if min_k > max_k {
            return Err(ClusterError::InvalidKRange { min_k, max_k });
        }
        Ok(Self {
            min_k,
            max_k,
            constraint,
            n_init: 10,
            max_iter: 75,
            tol: 1e-4,
            seed: 42,
            dba_max_iter: 10,
            precompute_matrix: true,
        })
    }

    /// Set the number of independent restarts per k value.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the maximum number of EM iterations per restart.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the maximum number of DBA iterations used when computing centroids.
    #[must_use]
    pub fn with_dba_max_iter(mut self, dba_max_iter: usize) -> Self {
        self.dba_max_iter = dba_max_iter;
        self
    }

    /// Return the minimum cluster count (inclusive).
    #[must_use]
    pub fn min_k(&self) -> usize {
        self.min_k
    }

    /// Return the maximum cluster count (inclusive).
    #[must_use]
    pub fn max_k(&self) -> usize {
        self.max_k
    }

    /// Return the band constraint used for DTW distance computation.
    #[must_use]
    pub fn constraint(&self) -> BandConstraint {
        self.constraint
    }

    /// Return the number of independent restarts per k value.
    #[must_use]
    pub fn n_init(&self) -> usize {
        self.n_init
    }

    /// Return the maximum number of EM iterations per restart.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the maximum number of DBA iterations used when computing centroids.
    #[must_use]
    pub fn dba_max_iter(&self) -> usize {
        self.dba_max_iter
    }

    /// Set whether to precompute the pairwise distance matrix for K-means++ initialization.
    ///
    /// When enabled, a single O(n^2) DTW pairwise computation is performed once and
    /// reused across all K-means restarts and k values, avoiding redundant distance
    /// calculations during initialization.
    #[must_use]
    pub fn with_precompute_matrix(mut self, precompute_matrix: bool) -> Self {
        self.precompute_matrix = precompute_matrix;
        self
    }

    /// Return whether pairwise distance matrix precomputation is enabled.
    #[must_use]
    pub fn precompute_matrix(&self) -> bool {
        self.precompute_matrix
    }

    /// Run K-means for each k in `[min_k, max_k]` and return the full inertia curve.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::TooFewSeries`] | `series.len() < max_k` |
    /// | [`ClusterError::EmptyCluster`] | A cluster becomes empty and cannot be rescued |
    /// | [`ClusterError::Dba`] | A DBA centroid update fails |
    pub fn fit(&self, series: &[TimeSeries]) -> Result<OptimizeResult, ClusterError> {
        let n = series.len();
        if n < self.max_k {
            return Err(ClusterError::TooFewSeries { n_series: n, k: self.max_k });
        }
        crate::kmeans::optimize(series, self)
    }
}

// ── MiniBatchConfig ───────────────────────────────────────────────────────────

/// Configuration for mini-batch K-means clustering.
///
/// Mini-batch K-means samples a random subset of the data at each iteration,
/// performs a standard assign step on the batch, and updates centroids with
/// a decaying learning rate. A final full-pass assignment is performed at
/// the end for accurate labels and inertia.
///
/// # Defaults
///
/// | Parameter    | Default |
/// |--------------|---------|
/// | `batch_size` | 256     |
/// | `n_init`     | 3       |
/// | `max_iter`   | 200     |
/// | `tol`        | 1e-4    |
/// | `seed`       | 42      |
#[derive(Debug, Clone)]
pub struct MiniBatchConfig {
    pub(crate) k: usize,
    pub(crate) constraint: BandConstraint,
    pub(crate) batch_size: usize,
    pub(crate) n_init: usize,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
    pub(crate) seed: u64,
}

impl MiniBatchConfig {
    /// Create a new mini-batch K-means configuration with the given cluster count
    /// and band constraint.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::InvalidK`] | `k` is zero |
    pub fn new(k: usize, constraint: BandConstraint) -> Result<Self, ClusterError> {
        if k == 0 {
            return Err(ClusterError::InvalidK { k });
        }
        Ok(Self {
            k,
            constraint,
            batch_size: 256,
            n_init: 3,
            max_iter: 200,
            tol: 1e-4,
            seed: 42,
        })
    }

    /// Set the mini-batch size. Larger batches give more stable centroid
    /// updates but reduce the speed advantage over full-batch K-means.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of independent restarts. Higher values reduce the risk of
    /// converging to a poor local minimum.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the maximum number of mini-batch iterations per restart.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance. Iteration stops when the total centroid
    /// shift across a mini-batch falls below this threshold.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed used for K-means++ initialization, batch sampling,
    /// and restart shuffling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Return the number of clusters.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the band constraint used for DTW distance computation.
    #[must_use]
    pub fn constraint(&self) -> BandConstraint {
        self.constraint
    }

    /// Return the mini-batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Return the number of independent restarts.
    #[must_use]
    pub fn n_init(&self) -> usize {
        self.n_init
    }

    /// Return the maximum number of mini-batch iterations per restart.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Return the random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Cluster `series` using this mini-batch configuration.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`ClusterError::TooFewSeries`] | `series.len() < k` |
    pub fn fit(&self, series: &[TimeSeries]) -> Result<KMeansResult, ClusterError> {
        let n = series.len();
        if n < self.k {
            return Err(ClusterError::TooFewSeries { n_series: n, k: self.k });
        }
        crate::minibatch::multi_restart_minibatch(series, self)
    }
}

#[cfg(test)]
mod tests {
    use narcissus_dtw::BandConstraint;

    use super::{InitStrategy, KMeansConfig, MiniBatchConfig, OptimizeConfig};
    use crate::error::ClusterError;

    #[test]
    fn new_valid_k() {
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained);
        assert!(cfg.is_ok());
        assert_eq!(cfg.unwrap().k(), 3);
    }

    #[test]
    fn new_k_zero() {
        let result = KMeansConfig::new(0, BandConstraint::Unconstrained);
        assert!(matches!(result, Err(ClusterError::InvalidK { k: 0 })));
    }

    #[test]
    fn builder_chaining() {
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(99);
        assert_eq!(cfg.n_init(), 5);
        assert_eq!(cfg.seed(), 99);
        assert_eq!(cfg.k(), 3);
    }

    #[test]
    fn optimize_valid() {
        let result = OptimizeConfig::new(2, 5, BandConstraint::Unconstrained);
        assert!(result.is_ok());
        let cfg = result.unwrap();
        assert_eq!(cfg.min_k(), 2);
        assert_eq!(cfg.max_k(), 5);
    }

    #[test]
    fn optimize_invalid_range() {
        let result = OptimizeConfig::new(5, 2, BandConstraint::Unconstrained);
        assert!(matches!(
            result,
            Err(ClusterError::InvalidKRange { min_k: 5, max_k: 2 })
        ));
    }

    #[test]
    fn optimize_min_k_zero() {
        let result = OptimizeConfig::new(0, 5, BandConstraint::Unconstrained);
        assert!(matches!(result, Err(ClusterError::InvalidK { k: 0 })));
    }

    #[test]
    fn defaults_are_correct() {
        let cfg = KMeansConfig::new(1, BandConstraint::Unconstrained).unwrap();
        assert_eq!(cfg.n_init(), 10);
        assert_eq!(cfg.max_iter(), 75);
        assert!((cfg.tol() - 1e-4).abs() < f64::EPSILON);
        assert_eq!(cfg.seed(), 42);
        assert_eq!(cfg.dba_max_iter(), 10);
    }

    #[test]
    fn use_elkan_default_false() {
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained).unwrap();
        assert!(!cfg.use_elkan(), "use_elkan should default to false");
    }

    #[test]
    fn precompute_matrix_default_true() {
        let cfg = OptimizeConfig::new(2, 5, BandConstraint::Unconstrained).unwrap();
        assert!(
            cfg.precompute_matrix(),
            "precompute_matrix should default to true"
        );
    }

    #[test]
    fn init_strategy_default_kpp() {
        let cfg = KMeansConfig::new(3, BandConstraint::Unconstrained).unwrap();
        assert_eq!(
            cfg.init_strategy(),
            InitStrategy::KMeansPlusPlus,
            "default init_strategy should be KMeansPlusPlus"
        );
    }

    #[test]
    fn minibatch_defaults() {
        let cfg = MiniBatchConfig::new(3, BandConstraint::Unconstrained).unwrap();
        assert_eq!(cfg.batch_size(), 256, "default batch_size should be 256");
        assert_eq!(cfg.n_init(), 3, "default n_init should be 3");
        assert_eq!(cfg.max_iter(), 200, "default max_iter should be 200");
        assert!((cfg.tol() - 1e-4).abs() < f64::EPSILON, "default tol should be 1e-4");
        assert_eq!(cfg.seed(), 42, "default seed should be 42");
    }

    #[test]
    fn minibatch_k_zero() {
        let result = MiniBatchConfig::new(0, BandConstraint::Unconstrained);
        assert!(
            matches!(result, Err(ClusterError::InvalidK { k: 0 })),
            "k=0 should return InvalidK error"
        );
    }
}
