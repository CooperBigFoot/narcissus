//! Stratified k-fold cross-validation for Random Forest.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use tracing::{info, instrument};

use crate::config::RandomForestConfig;
use crate::confusion::ConfusionMatrix;
use crate::error::RfError;
use crate::importance::{RankedFeature, aggregate_importances};

/// Cross-validation configuration.
///
/// Construct via [`CrossValidation::new`], then chain `with_seed` if desired.
#[derive(Debug, Clone)]
pub struct CrossValidation {
    n_folds: usize,
    seed: u64,
}

/// Results of stratified k-fold cross-validation.
#[derive(Debug)]
pub struct CrossValidationResult {
    /// Accuracy for each fold.
    pub fold_accuracies: Vec<f64>,
    /// Aggregated confusion matrix (summed across all folds).
    pub confusion_matrix: ConfusionMatrix,
    /// Mean accuracy across folds.
    pub mean_accuracy: f64,
    /// Standard deviation of fold accuracies.
    pub std_accuracy: f64,
    /// Averaged feature importances across all folds.
    pub feature_importances: Vec<RankedFeature>,
    /// Number of folds.
    pub n_folds: usize,
    /// Total number of samples.
    pub n_samples: usize,
    /// Number of features.
    pub n_features: usize,
    /// Number of classes.
    pub n_classes: usize,
}

impl CrossValidation {
    /// Create a new cross-validation config with the given number of folds.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::InvalidFoldCount`] if `n_folds` < 2.
    pub fn new(n_folds: usize) -> Result<Self, RfError> {
        if n_folds < 2 {
            return Err(RfError::InvalidFoldCount { n_folds });
        }
        Ok(Self { n_folds, seed: 42 })
    }

    /// Set the random seed for fold shuffling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run stratified k-fold cross-validation.
    ///
    /// Splits the data into `n_folds` folds with approximately equal class
    /// distribution in each fold. Each fold trains an RF on the remaining
    /// folds and evaluates on the held-out fold.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`RfError::EmptyDataset`] | Zero samples |
    /// | [`RfError::TooFewSamplesForFolds`] | A class has fewer samples than folds |
    /// | Other RF errors | From underlying training |
    #[instrument(skip_all, fields(n_folds = self.n_folds, n_samples = features.len()))]
    pub fn evaluate(
        &self,
        config: &RandomForestConfig,
        features: &[Vec<f64>],
        labels: &[usize],
        feature_names: &[String],
    ) -> Result<CrossValidationResult, RfError> {
        if features.is_empty() {
            return Err(RfError::EmptyDataset);
        }

        let n_samples = features.len();
        let n_features = features[0].len();
        let n_classes = labels.iter().max().copied().unwrap_or(0) + 1;

        // Create stratified folds.
        let fold_assignments = self.stratified_split(labels, n_classes)?;

        let mut fold_accuracies = Vec::with_capacity(self.n_folds);
        let mut all_true_labels = Vec::new();
        let mut all_predicted = Vec::new();
        let mut all_importances: Vec<Vec<f64>> = Vec::new();

        for fold in 0..self.n_folds {
            // Split into train/test for this fold.
            let mut train_features = Vec::new();
            let mut train_labels = Vec::new();
            let mut test_features = Vec::new();
            let mut test_labels = Vec::new();

            for (i, &assigned_fold) in fold_assignments.iter().enumerate() {
                if assigned_fold == fold {
                    test_features.push(features[i].clone());
                    test_labels.push(labels[i]);
                } else {
                    train_features.push(features[i].clone());
                    train_labels.push(labels[i]);
                }
            }

            // Clone config and override seed so each fold trains with different randomness.
            let fold_config = config.clone().with_seed(config.seed.wrapping_add(fold as u64));

            let result = fold_config.fit(&train_features, &train_labels, feature_names)?;

            // Predict on test fold.
            let predictions = result.forest().predict_batch(&test_features)?;

            // Compute fold accuracy.
            let correct = predictions
                .iter()
                .zip(&test_labels)
                .filter(|&(&p, &l)| p == l)
                .count();
            let fold_accuracy = correct as f64 / test_labels.len() as f64;
            fold_accuracies.push(fold_accuracy);

            info!(fold, accuracy = fold_accuracy, "fold completed");

            // Accumulate for overall confusion matrix.
            all_true_labels.extend_from_slice(&test_labels);
            all_predicted.extend_from_slice(&predictions);

            // Accumulate per-tree importances from this fold's forest.
            let forest = result.forest();
            let tree_importances: Vec<Vec<f64>> =
                forest.trees.iter().map(|t| t.feature_importances()).collect();
            all_importances.extend(tree_importances);
        }

        // Aggregate results.
        let mean_accuracy = fold_accuracies.iter().sum::<f64>() / self.n_folds as f64;
        let std_accuracy = {
            let variance = fold_accuracies
                .iter()
                .map(|&a| (a - mean_accuracy).powi(2))
                .sum::<f64>()
                / self.n_folds as f64;
            variance.sqrt()
        };

        let confusion_matrix =
            ConfusionMatrix::from_labels(&all_true_labels, &all_predicted, n_classes)?;

        let feature_importances = aggregate_importances(&all_importances, feature_names);

        info!(
            mean_accuracy,
            std_accuracy,
            "cross-validation complete"
        );

        Ok(CrossValidationResult {
            fold_accuracies,
            confusion_matrix,
            mean_accuracy,
            std_accuracy,
            feature_importances,
            n_folds: self.n_folds,
            n_samples,
            n_features,
            n_classes,
        })
    }

    /// Create stratified fold assignments.
    ///
    /// Groups samples by class, shuffles within each class, then
    /// round-robins across folds so each fold gets approximately
    /// equal representation of each class.
    fn stratified_split(&self, labels: &[usize], n_classes: usize) -> Result<Vec<usize>, RfError> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        // Group indices by class.
        let mut class_indices: Vec<Vec<usize>> = vec![vec![]; n_classes];
        for (i, &label) in labels.iter().enumerate() {
            class_indices[label].push(i);
        }

        // Validate: each class needs at least n_folds samples.
        for (class, indices) in class_indices.iter().enumerate() {
            if !indices.is_empty() && indices.len() < self.n_folds {
                return Err(RfError::TooFewSamplesForFolds {
                    class,
                    count: indices.len(),
                    n_folds: self.n_folds,
                });
            }
        }

        // Shuffle within each class and assign folds round-robin.
        let mut fold_assignments = vec![0usize; labels.len()];

        for indices in &mut class_indices {
            indices.shuffle(&mut rng);
            for (j, &idx) in indices.iter().enumerate() {
                fold_assignments[idx] = j % self.n_folds;
            }
        }

        Ok(fold_assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MaxFeatures;

    fn make_separable_data() -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        // 3 classes, 30 samples each
        for i in 0..30 {
            features.push(vec![i as f64 * 0.1, 0.5]);
            labels.push(0);
        }
        for i in 0..30 {
            features.push(vec![10.0 + i as f64 * 0.1, 0.5]);
            labels.push(1);
        }
        for i in 0..30 {
            features.push(vec![20.0 + i as f64 * 0.1, 0.5]);
            labels.push(2);
        }
        let names = vec!["x".to_string(), "y".to_string()];
        (features, labels, names)
    }

    #[test]
    fn five_fold_separable_accuracy() {
        let (features, labels, names) = make_separable_data();
        let rf_config = RandomForestConfig::new(20)
            .unwrap()
            .with_max_features(MaxFeatures::All)
            .with_seed(42);
        let cv = CrossValidation::new(5).unwrap().with_seed(42);
        let result = cv.evaluate(&rf_config, &features, &labels, &names).unwrap();

        assert!(
            result.mean_accuracy > 0.8,
            "mean_accuracy = {}",
            result.mean_accuracy
        );
        assert_eq!(result.fold_accuracies.len(), 5);
        assert_eq!(result.n_folds, 5);
        assert_eq!(result.n_samples, 90);
    }

    #[test]
    fn fold_count_matches() {
        let (features, labels, names) = make_separable_data();
        let rf_config = RandomForestConfig::new(5).unwrap().with_seed(42);
        let cv = CrossValidation::new(3).unwrap();
        let result = cv.evaluate(&rf_config, &features, &labels, &names).unwrap();
        assert_eq!(result.fold_accuracies.len(), 3);
    }

    #[test]
    fn confusion_matrix_dimensions() {
        let (features, labels, names) = make_separable_data();
        let rf_config = RandomForestConfig::new(10).unwrap().with_seed(42);
        let cv = CrossValidation::new(3).unwrap();
        let result = cv.evaluate(&rf_config, &features, &labels, &names).unwrap();
        assert_eq!(result.confusion_matrix.n_classes(), 3);
        assert_eq!(result.confusion_matrix.as_rows().len(), 3);
    }

    #[test]
    fn feature_importances_sum_to_one() {
        let (features, labels, names) = make_separable_data();
        let rf_config = RandomForestConfig::new(10).unwrap().with_seed(42);
        let cv = CrossValidation::new(3).unwrap();
        let result = cv.evaluate(&rf_config, &features, &labels, &names).unwrap();
        let total: f64 = result.feature_importances.iter().map(|f| f.importance).sum();
        assert!((total - 1.0).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn invalid_fold_count() {
        assert!(CrossValidation::new(0).is_err());
        assert!(CrossValidation::new(1).is_err());
    }

    #[test]
    fn too_few_samples_for_folds() {
        // 2 samples in class 0, but requesting 5 folds
        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ];
        let labels = vec![0, 0, 1, 1, 1];
        let names = vec!["x".to_string()];
        let rf_config = RandomForestConfig::new(5).unwrap();
        let cv = CrossValidation::new(5).unwrap();
        let err = cv
            .evaluate(&rf_config, &features, &labels, &names)
            .unwrap_err();
        assert!(matches!(
            err,
            RfError::TooFewSamplesForFolds {
                class: 0,
                count: 2,
                n_folds: 5
            }
        ));
    }
}
