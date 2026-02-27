//! Prediction methods for the Random Forest ensemble.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::error::RfError;
use crate::forest::RandomForest;

/// Class probability distribution from a prediction.
#[derive(Debug, Clone)]
pub struct ClassDistribution {
    probs: Vec<f64>,
}

impl ClassDistribution {
    /// Create a new class distribution.
    pub(crate) fn new(probs: Vec<f64>) -> Self {
        Self { probs }
    }

    /// Return the predicted class (argmax of probabilities).
    #[must_use]
    pub fn predicted_class(&self) -> usize {
        self.probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Return the top-k classes sorted by descending probability.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self.probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(k);
        indexed
    }

    /// Return the probability distribution as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.probs
    }
}

impl RandomForest {
    /// Predict the class label for a single sample.
    ///
    /// Returns the argmax of the averaged probability distribution.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] when `sample.len() != n_features`.
    pub fn predict(&self, sample: &[f64]) -> Result<usize, RfError> {
        Ok(self.predict_proba(sample)?.predicted_class())
    }

    /// Return the averaged class probability distribution for a single sample.
    ///
    /// Averages the leaf distributions from all trees.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] when `sample.len() != n_features`.
    pub fn predict_proba(&self, sample: &[f64]) -> Result<ClassDistribution, RfError> {
        if sample.len() != self.n_features {
            return Err(RfError::PredictionFeatureMismatch {
                expected: self.n_features,
                got: sample.len(),
            });
        }

        let mut avg = vec![0.0f64; self.n_classes];
        for tree in &self.trees {
            let proba = tree.predict_proba(sample)?;
            for (i, p) in proba.iter().enumerate() {
                avg[i] += p;
            }
        }
        let n = self.trees.len() as f64;
        avg.iter_mut().for_each(|v| *v /= n);

        Ok(ClassDistribution::new(avg))
    }

    /// Predict class labels for a batch of samples in parallel.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] if any sample has the wrong feature count.
    pub fn predict_batch(&self, features: &[Vec<f64>]) -> Result<Vec<usize>, RfError> {
        features
            .into_par_iter()
            .map(|sample| self.predict(sample))
            .collect()
    }

    /// Return probability distributions for a batch of samples in parallel.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] if any sample has the wrong feature count.
    pub fn predict_proba_batch(
        &self,
        features: &[Vec<f64>],
    ) -> Result<Vec<ClassDistribution>, RfError> {
        features
            .into_par_iter()
            .map(|sample| self.predict_proba(sample))
            .collect()
    }

    /// Return the number of features this forest was trained on.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Return the number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Return the number of trees in the ensemble.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Return the feature names.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}
