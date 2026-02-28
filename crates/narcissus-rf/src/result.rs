//! Training result types for Random Forest.

use crate::forest::RandomForest;
use crate::importance::RankedFeature;
use crate::oob::OobScore;
use crate::perm_importance::{PermutationImportance, compute_permutation_importance};

/// Metadata about the training run.
#[derive(Debug, Clone)]
pub struct TrainingMetadata {
    /// Number of trees trained.
    pub n_trees: usize,
    /// Number of features in the dataset.
    pub n_features: usize,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Number of training samples.
    pub n_samples: usize,
    /// Resolved max_features value used.
    pub max_features_resolved: usize,
}

/// Result of Random Forest training.
///
/// Contains the fitted forest, feature importances, optional OOB score,
/// per-tree OOB indices, and training metadata.
#[derive(Debug)]
pub struct RandomForestResult {
    forest: RandomForest,
    importances: Vec<RankedFeature>,
    oob_score: Option<OobScore>,
    oob_indices_per_tree: Vec<Vec<usize>>,
    metadata: TrainingMetadata,
}

impl RandomForestResult {
    /// Create a new training result.
    pub(crate) fn new(
        forest: RandomForest,
        importances: Vec<RankedFeature>,
        oob_score: Option<OobScore>,
        oob_indices_per_tree: Vec<Vec<usize>>,
        metadata: TrainingMetadata,
    ) -> Self {
        Self {
            forest,
            importances,
            oob_score,
            oob_indices_per_tree,
            metadata,
        }
    }

    /// Borrow the fitted forest.
    #[must_use]
    pub fn forest(&self) -> &RandomForest {
        &self.forest
    }

    /// Consume the result and return the fitted forest.
    #[must_use]
    pub fn into_forest(self) -> RandomForest {
        self.forest
    }

    /// Return the ranked feature importances.
    #[must_use]
    pub fn importances(&self) -> &[RankedFeature] {
        &self.importances
    }

    /// Return the OOB score, if computed.
    #[must_use]
    pub fn oob_score(&self) -> Option<&OobScore> {
        self.oob_score.as_ref()
    }

    /// Return training metadata.
    #[must_use]
    pub fn metadata(&self) -> &TrainingMetadata {
        &self.metadata
    }

    /// Return the per-tree OOB sample indices.
    #[must_use]
    pub fn oob_indices_per_tree(&self) -> &[Vec<usize>] {
        &self.oob_indices_per_tree
    }

    /// Compute permutation feature importance using OOB samples.
    ///
    /// Requires the original training data (features and labels) since they
    /// are not stored in the result.
    ///
    /// # Panics
    ///
    /// Panics if `features` or `labels` dimensions don't match the training data.
    #[must_use]
    pub fn permutation_importances(
        &self,
        features: &[Vec<f64>],
        labels: &[usize],
        seed: u64,
    ) -> Vec<PermutationImportance> {
        compute_permutation_importance(
            &self.forest,
            features,
            labels,
            &self.oob_indices_per_tree,
            &self.forest.feature_names,
            seed,
        )
    }
}
