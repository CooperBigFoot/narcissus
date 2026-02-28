//! Configuration builder for Random Forest training.

use crate::error::RfError;
use crate::result::RandomForestResult;
use crate::split::{SplitCriterion, SplitMethod};

/// Strategy for determining the number of features to consider at each split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxFeatures {
    /// Square root of total features.
    Sqrt,
    /// Log base 2 of total features.
    Log2,
    /// A fraction of total features (must be in (0.0, 1.0]).
    Fraction(f64),
    /// A fixed count.
    Fixed(usize),
    /// All features (no subsampling).
    All,
}

/// Whether to compute out-of-bag evaluation during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OobMode {
    /// Compute OOB accuracy and confusion matrix.
    Enabled,
    /// Skip OOB evaluation.
    Disabled,
}

/// Configuration for Random Forest training.
///
/// Construct via [`RandomForestConfig::new`], then chain `with_*` methods.
///
/// # Defaults
///
/// | Parameter            | Default     |
/// |----------------------|-------------|
/// | `max_features`       | `Sqrt`      |
/// | `max_depth`          | `None`      |
/// | `min_samples_split`  | 2           |
/// | `min_samples_leaf`   | 1           |
/// | `criterion`          | `Gini`      |
/// | `split_method`       | `Exact`     |
/// | `seed`               | 42          |
/// | `oob_mode`           | `Disabled`  |
/// | `bootstrap_fraction` | 1.0         |
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    pub(crate) n_trees: usize,
    pub(crate) max_features: MaxFeatures,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) criterion: SplitCriterion,
    pub(crate) split_method: SplitMethod,
    pub(crate) seed: u64,
    pub(crate) oob_mode: OobMode,
    pub(crate) bootstrap_fraction: f64,
}

impl RandomForestConfig {
    /// Create a new config with the given number of trees.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::InvalidTreeCount`] if `n_trees` is zero.
    pub fn new(n_trees: usize) -> Result<Self, RfError> {
        if n_trees == 0 {
            return Err(RfError::InvalidTreeCount { n_trees });
        }
        Ok(Self {
            n_trees,
            max_features: MaxFeatures::Sqrt,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            split_method: SplitMethod::Exact,
            seed: 42,
            oob_mode: OobMode::Disabled,
            bootstrap_fraction: 1.0,
        })
    }

    // --- Setters ---

    /// Set the max features strategy.
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the maximum tree depth. `None` means unlimited.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to attempt a split.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in each leaf after a split.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the split quality criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the split-finding strategy.
    #[must_use]
    pub fn with_split_method(mut self, split_method: SplitMethod) -> Self {
        self.split_method = split_method;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the OOB evaluation mode.
    #[must_use]
    pub fn with_oob_mode(mut self, oob_mode: OobMode) -> Self {
        self.oob_mode = oob_mode;
        self
    }

    /// Set the bootstrap fraction (proportion of samples drawn per tree).
    #[must_use]
    pub fn with_bootstrap_fraction(mut self, bootstrap_fraction: f64) -> Self {
        self.bootstrap_fraction = bootstrap_fraction;
        self
    }

    // --- Getters ---

    /// Return the number of trees.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// Return the max features strategy.
    #[must_use]
    pub fn max_features(&self) -> MaxFeatures {
        self.max_features
    }

    /// Return the maximum depth limit, if any.
    #[must_use]
    pub fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Return the minimum samples required to split a node.
    #[must_use]
    pub fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Return the minimum samples required in each leaf.
    #[must_use]
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Return the split criterion.
    #[must_use]
    pub fn criterion(&self) -> SplitCriterion {
        self.criterion
    }

    /// Return the split-finding strategy.
    #[must_use]
    pub fn split_method(&self) -> SplitMethod {
        self.split_method
    }

    /// Return the random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the OOB evaluation mode.
    #[must_use]
    pub fn oob_mode(&self) -> OobMode {
        self.oob_mode
    }

    /// Return the bootstrap fraction.
    #[must_use]
    pub fn bootstrap_fraction(&self) -> f64 {
        self.bootstrap_fraction
    }

    /// Train a Random Forest on the provided dataset.
    ///
    /// `features[sample_idx][feature_idx]` — row-major layout.
    /// `labels[sample_idx]` — class labels (zero-based).
    /// `feature_names` — names for each feature column.
    ///
    /// # Errors
    ///
    /// | Variant                               | When                                              |
    /// |---------------------------------------|---------------------------------------------------|
    /// | [`RfError::InvalidTreeCount`]         | `n_trees` is zero                                 |
    /// | [`RfError::EmptyDataset`]             | `features` is empty                               |
    /// | [`RfError::ZeroFeatures`]             | rows have zero feature columns                    |
    /// | [`RfError::FeatureCountMismatch`]     | rows have inconsistent lengths                    |
    /// | [`RfError::NonFiniteValue`]           | any value is NaN or infinite                      |
    /// | [`RfError::InvalidMaxFeatures`]       | resolved max_features is outside [1, n_features]  |
    /// | [`RfError::InvalidBootstrapFraction`] | bootstrap_fraction is not in (0.0, 1.0]           |
    /// | [`RfError::OobEvaluationFailed`]      | OOB enabled but no sample has any OOB tree        |
    pub fn fit(
        &self,
        features: &[Vec<f64>],
        labels: &[usize],
        feature_names: &[String],
    ) -> Result<RandomForestResult, RfError> {
        crate::forest::train(self, features, labels, feature_names)
    }
}
