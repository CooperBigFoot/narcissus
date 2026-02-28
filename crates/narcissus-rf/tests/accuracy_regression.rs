//! Accuracy regression tests for narcissus-rf.
//!
//! These tests verify that algorithmic changes do not degrade Random Forest
//! classification accuracy on a deterministic synthetic dataset.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use narcissus_rf::{CrossValidation, OobMode, RandomForestConfig};

// ---------------------------------------------------------------------------
// Helper: deterministic synthetic classification dataset
// ---------------------------------------------------------------------------

/// Generate a 300-sample, 10-feature, 3-class classification dataset.
///
/// Features 0-2 are informative (class * 3.0 + noise in [0, 0.5]).
/// Features 3-9 are pure noise in [0, 0.5].
/// Samples are assigned round-robin across classes.
fn make_classification() -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_samples = 300;
    let n_features = 10;
    let n_classes = 3;

    let mut features = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let class = i % n_classes;
        labels.push(class);
        let row: Vec<f64> = (0..n_features)
            .map(|f| {
                let base = if f < 3 { class as f64 * 3.0 } else { 0.0 };
                base + rng.r#gen::<f64>() * 0.5
            })
            .collect();
        features.push(row);
    }
    let names: Vec<String> = (0..n_features).map(|f| format!("f{f}")).collect();
    (features, labels, names)
}

// ---------------------------------------------------------------------------
// a) cv_accuracy_above_threshold
// ---------------------------------------------------------------------------

/// 5-fold cross-validation mean accuracy must exceed 0.85 on the synthetic dataset.
///
/// Reference: observed mean_accuracy = 1.0 with seed=42, 100 trees.
#[test]
fn cv_accuracy_above_threshold() {
    let (features, labels, names) = make_classification();
    let rf_config = RandomForestConfig::new(100).unwrap().with_seed(42);
    let cv = CrossValidation::new(5).unwrap().with_seed(42);
    let result = cv.evaluate(&rf_config, &features, &labels, &names).unwrap();

    assert!(
        result.mean_accuracy > 0.85,
        "cv mean_accuracy {} <= 0.85",
        result.mean_accuracy
    );
}

// ---------------------------------------------------------------------------
// b) oob_accuracy_above_threshold
// ---------------------------------------------------------------------------

/// OOB accuracy with 100 trees must exceed 0.80.
///
/// Reference: observed oob_accuracy = 1.0 with seed=42, 100 trees.
#[test]
fn oob_accuracy_above_threshold() {
    let (features, labels, names) = make_classification();
    let rf_config = RandomForestConfig::new(100)
        .unwrap()
        .with_seed(42)
        .with_oob_mode(OobMode::Enabled);
    let result = rf_config.fit(&features, &labels, &names).unwrap();

    let oob = result.oob_score().expect("OOB score must be computed when OobMode::Enabled");
    assert!(
        oob.accuracy > 0.80,
        "oob_accuracy {} <= 0.80",
        oob.accuracy
    );
}

// ---------------------------------------------------------------------------
// c) top_features_are_informative
// ---------------------------------------------------------------------------

/// The top 3 features by importance must include at least 2 of f0, f1, f2.
///
/// Features f0, f1, f2 are the informative ones in the synthetic dataset;
/// f3-f9 are pure noise. A correctly functioning forest must rank informative
/// features above noise features.
#[test]
fn top_features_are_informative() {
    let (features, labels, names) = make_classification();
    let rf_config = RandomForestConfig::new(100).unwrap().with_seed(42);
    let result = rf_config.fit(&features, &labels, &names).unwrap();

    let informative: std::collections::HashSet<&str> = ["f0", "f1", "f2"].iter().copied().collect();

    let top3_names: Vec<&str> = result
        .importances()
        .iter()
        .take(3)
        .map(|f| f.name.as_str())
        .collect();

    let informative_in_top3 = top3_names.iter().filter(|&&n| informative.contains(n)).count();

    assert!(
        informative_in_top3 >= 2,
        "only {informative_in_top3}/3 of top-3 features are informative; top-3: {top3_names:?}"
    );
}

// ---------------------------------------------------------------------------
// d) deterministic_predictions
// ---------------------------------------------------------------------------

/// Same config and seed must produce identical predictions across two independent runs.
#[test]
fn deterministic_predictions() {
    let (features, labels, names) = make_classification();
    let rf_config = RandomForestConfig::new(100).unwrap().with_seed(42);

    let result1 = rf_config.fit(&features, &labels, &names).unwrap();
    let result2 = rf_config.fit(&features, &labels, &names).unwrap();

    let preds1 = result1.forest().predict_batch(&features).unwrap();
    let preds2 = result2.forest().predict_batch(&features).unwrap();

    assert_eq!(
        preds1, preds2,
        "predictions differ across runs with the same seed"
    );
}

// ---------------------------------------------------------------------------
// e) prediction_accuracy_on_training_data
// ---------------------------------------------------------------------------

/// Training accuracy with 100 trees must exceed 0.95 (RF should memorize training data).
///
/// Reference: observed training accuracy = 1.0 with seed=42, 100 trees.
#[test]
fn prediction_accuracy_on_training_data() {
    let (features, labels, names) = make_classification();
    let rf_config = RandomForestConfig::new(100).unwrap().with_seed(42);
    let result = rf_config.fit(&features, &labels, &names).unwrap();

    let predictions = result.forest().predict_batch(&features).unwrap();
    let correct = predictions
        .iter()
        .zip(&labels)
        .filter(|&(&p, &l)| p == l)
        .count();
    let accuracy = correct as f64 / labels.len() as f64;

    assert!(
        accuracy > 0.95,
        "training accuracy {} <= 0.95",
        accuracy
    );
}
