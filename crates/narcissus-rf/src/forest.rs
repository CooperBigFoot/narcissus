//! Random Forest training with parallel tree construction.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::{debug, info, instrument};

use crate::config::{MaxFeatures, OobMode, RandomForestConfig};
use crate::error::RfError;
use crate::importance::aggregate_importances;
use crate::oob::compute_oob;
use crate::result::{RandomForestResult, TrainingMetadata};
use crate::tree::{DecisionTree, DecisionTreeConfig};

/// A fitted Random Forest ensemble.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomForest {
    pub(crate) trees: Vec<DecisionTree>,
    pub(crate) n_features: usize,
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
}

/// Resolve `MaxFeatures` to a concrete count.
pub(crate) fn resolve_max_features(
    max_features: MaxFeatures,
    n_features: usize,
) -> Result<usize, RfError> {
    let resolved = match max_features {
        MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
        MaxFeatures::Log2 => (n_features as f64).log2().ceil().max(1.0) as usize,
        MaxFeatures::Fraction(f) => (n_features as f64 * f).ceil() as usize,
        MaxFeatures::Fixed(n) => n,
        MaxFeatures::All => n_features,
    };
    if resolved == 0 || resolved > n_features {
        return Err(RfError::InvalidMaxFeatures {
            max_features: resolved,
            n_features,
        });
    }
    Ok(resolved)
}

/// Generate a bootstrap sample and the out-of-bag indices.
fn bootstrap_sample(
    n_samples: usize,
    draw_count: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut in_bag = vec![false; n_samples];
    let mut bootstrap_indices = Vec::with_capacity(draw_count);
    for _ in 0..draw_count {
        let idx = rng.gen_range(0..n_samples);
        bootstrap_indices.push(idx);
        in_bag[idx] = true;
    }
    let oob_indices: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();
    (bootstrap_indices, oob_indices)
}

/// Train the Random Forest ensemble.
#[instrument(skip_all, fields(n_trees = config.n_trees, n_samples = features.len()))]
pub(crate) fn train(
    config: &RandomForestConfig,
    features: &[Vec<f64>],
    labels: &[usize],
    feature_names: &[String],
) -> Result<RandomForestResult, RfError> {
    // --- Validate inputs ---
    if features.is_empty() {
        return Err(RfError::EmptyDataset);
    }
    let n_samples = features.len();
    let n_features = features[0].len();
    if n_features == 0 {
        return Err(RfError::ZeroFeatures);
    }
    for (sample_index, row) in features.iter().enumerate() {
        if row.len() != n_features {
            return Err(RfError::FeatureCountMismatch {
                expected: n_features,
                got: row.len(),
                sample_index,
            });
        }
        for (feature_index, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(RfError::NonFiniteValue {
                    sample_index,
                    feature_index,
                });
            }
        }
    }

    // --- Validate config ---
    let max_features_resolved = resolve_max_features(config.max_features, n_features)?;

    if config.bootstrap_fraction <= 0.0 || config.bootstrap_fraction > 1.0 {
        return Err(RfError::InvalidBootstrapFraction {
            fraction: config.bootstrap_fraction,
        });
    }

    let n_classes = labels.iter().max().copied().unwrap_or(0) + 1;
    let draw_count = ((n_samples as f64) * config.bootstrap_fraction).ceil() as usize;

    info!(
        n_trees = config.n_trees,
        n_samples,
        n_features,
        n_classes,
        max_features = max_features_resolved,
        draw_count,
        "training random forest"
    );

    // Generate per-tree seeds from master RNG.
    let mut master_rng = ChaCha8Rng::seed_from_u64(config.seed);
    let tree_seeds: Vec<u64> = (0..config.n_trees).map(|_| master_rng.r#gen()).collect();

    // Capture config fields needed in closure (avoids borrowing config across thread boundary).
    let criterion = config.criterion;
    let split_method = config.split_method;
    let max_depth = config.max_depth;
    let min_samples_split = config.min_samples_split;
    let min_samples_leaf = config.min_samples_leaf;

    // Parallel tree training.
    let tree_results: Vec<(DecisionTree, Vec<usize>)> = tree_seeds
        .into_par_iter()
        .map(|seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (bootstrap_indices, oob_indices) =
                bootstrap_sample(n_samples, draw_count, &mut rng);

            // Build bootstrap dataset: row-major features.
            let boot_features: Vec<Vec<f64>> = bootstrap_indices
                .iter()
                .map(|&i| features[i].clone())
                .collect();
            let boot_labels: Vec<usize> =
                bootstrap_indices.iter().map(|&i| labels[i]).collect();

            let tree_config = DecisionTreeConfig::new()
                .with_criterion(criterion)
                .with_split_method(split_method)
                .with_max_depth(max_depth)
                .with_min_samples_split(min_samples_split)
                .with_min_samples_leaf(min_samples_leaf)
                .with_max_features(Some(max_features_resolved))
                .with_seed(rng.r#gen());

            // All inputs are pre-validated — fit cannot fail on data errors.
            let tree = tree_config
                .fit(&boot_features, &boot_labels)
                .expect("tree fit should not fail on pre-validated data");

            (tree, oob_indices)
        })
        .collect();

    let mut trees = Vec::with_capacity(config.n_trees);
    let mut oob_indices_per_tree = Vec::with_capacity(config.n_trees);
    for (tree, oob) in tree_results {
        trees.push(tree);
        oob_indices_per_tree.push(oob);
    }

    // Aggregate feature importances.
    let per_tree_importances: Vec<Vec<f64>> =
        trees.iter().map(|t| t.feature_importances()).collect();
    let importances = aggregate_importances(&per_tree_importances, feature_names);

    debug!(n_trees_trained = trees.len(), "tree training complete");

    // OOB evaluation.
    let oob_score = if config.oob_mode == OobMode::Enabled {
        Some(compute_oob(
            &trees,
            features,
            labels,
            n_features,
            n_classes,
            &oob_indices_per_tree,
        )?)
    } else {
        None
    };

    let forest = RandomForest {
        trees,
        n_features,
        n_classes,
        feature_names: feature_names.to_vec(),
    };

    let metadata = TrainingMetadata {
        n_trees: config.n_trees,
        n_features,
        n_classes,
        n_samples,
        max_features_resolved,
    };

    info!(
        oob_accuracy = oob_score.as_ref().map(|s| s.accuracy),
        "random forest training complete"
    );

    Ok(RandomForestResult::new(forest, importances, oob_score, oob_indices_per_tree, metadata))
}

#[cfg(test)]
mod tests {
    use crate::config::{MaxFeatures, OobMode, RandomForestConfig};
    use crate::split::SplitMethod;

    /// Generate a simple 3-class separable dataset.
    fn make_separable_data() -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        // Class 0: x in [0, 3]
        for i in 0..20 {
            features.push(vec![i as f64 * 0.15, 0.5]);
            labels.push(0);
        }
        // Class 1: x in [10, 13]
        for i in 0..20 {
            features.push(vec![10.0 + i as f64 * 0.15, 0.5]);
            labels.push(1);
        }
        // Class 2: x in [20, 23]
        for i in 0..20 {
            features.push(vec![20.0 + i as f64 * 0.15, 0.5]);
            labels.push(2);
        }
        let names = vec!["x".to_string(), "y".to_string()];
        (features, labels, names)
    }

    #[test]
    fn three_class_separable_accuracy() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_max_features(MaxFeatures::All)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();

        // Predict on training data — should get high accuracy.
        let predictions = result.forest().predict_batch(&features).unwrap();
        let correct = predictions
            .iter()
            .zip(&labels)
            .filter(|&(&p, &l)| p == l)
            .count();
        let accuracy = correct as f64 / labels.len() as f64;
        assert!(accuracy > 0.9, "accuracy = {accuracy}");
    }

    #[test]
    fn oob_score_computed() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_oob_mode(OobMode::Enabled)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();

        let oob = result.oob_score().expect("OOB should be computed");
        assert!(oob.accuracy > 0.8, "oob accuracy = {}", oob.accuracy);
        assert!(oob.n_oob_samples > 0);
    }

    #[test]
    fn feature_importances_sum_to_one() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(20).unwrap().with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();

        let total: f64 = result.importances().iter().map(|f| f.importance).sum();
        assert!((total - 1.0).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn deterministic_with_same_seed() {
        let (features, labels, names) = make_separable_data();
        let result1 = RandomForestConfig::new(10)
            .unwrap()
            .with_seed(99)
            .fit(&features, &labels, &names)
            .unwrap();
        let result2 = RandomForestConfig::new(10)
            .unwrap()
            .with_seed(99)
            .fit(&features, &labels, &names)
            .unwrap();

        let preds1 = result1.forest().predict_batch(&features).unwrap();
        let preds2 = result2.forest().predict_batch(&features).unwrap();
        assert_eq!(preds1, preds2);
    }

    #[test]
    fn predict_proba_batch_matches_individual() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(10).unwrap().with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();
        let forest = result.forest();

        let batch = forest.predict_proba_batch(&features).unwrap();
        for (i, sample) in features.iter().enumerate() {
            let single = forest.predict_proba(sample).unwrap();
            assert_eq!(batch[i].as_slice(), single.as_slice());
        }
    }

    #[test]
    fn invalid_tree_count_error() {
        assert!(RandomForestConfig::new(0).is_err());
    }

    #[test]
    fn empty_dataset_error() {
        let config = RandomForestConfig::new(10).unwrap();
        let err = config.fit(&[], &[], &[]).unwrap_err();
        assert!(matches!(err, crate::RfError::EmptyDataset));
    }

    #[test]
    fn extra_trees_three_class_accuracy() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_max_features(MaxFeatures::All)
            .with_split_method(SplitMethod::ExtraTrees)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();

        let predictions = result.forest().predict_batch(&features).unwrap();
        let correct = predictions
            .iter()
            .zip(&labels)
            .filter(|&(&p, &l)| p == l)
            .count();
        let accuracy = correct as f64 / labels.len() as f64;
        assert!(accuracy > 0.85, "extra-trees accuracy = {accuracy}");
    }

    #[test]
    fn histogram_three_class_accuracy() {
        let (features, labels, names) = make_separable_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_max_features(MaxFeatures::All)
            .with_split_method(SplitMethod::Histogram { n_bins: 32 })
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();

        let predictions = result.forest().predict_batch(&features).unwrap();
        let correct = predictions
            .iter()
            .zip(&labels)
            .filter(|&(&p, &l)| p == l)
            .count();
        let accuracy = correct as f64 / labels.len() as f64;
        assert!(accuracy > 0.85, "histogram RF accuracy = {accuracy}");
    }
}
