//! Model serialization and deserialization via bincode.

use std::path::Path;

use tracing::{debug, info, instrument};

use crate::error::RfError;
use crate::forest::RandomForest;

/// Current binary format version.
const FORMAT_VERSION: u32 = 1;

/// Versioned envelope for the serialized model.
#[derive(serde::Serialize, serde::Deserialize)]
struct ModelEnvelope {
    /// Format version for compatibility checking.
    format_version: u32,
    /// Number of trees in the forest.
    n_trees: usize,
    /// Number of features the model was trained on.
    n_features: usize,
    /// Number of classes.
    n_classes: usize,
    /// Feature column names.
    feature_names: Vec<String>,
    /// The serialized forest.
    forest: RandomForest,
}

impl RandomForest {
    /// Save the model to a binary file.
    ///
    /// Uses bincode encoding wrapped in a versioned envelope for
    /// forward-compatibility checking.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`RfError::SerializeModel`] | bincode encoding failed |
    /// | [`RfError::WriteModel`] | file write failed |
    #[instrument(skip(self), fields(path = %path.as_ref().display()))]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), RfError> {
        let path = path.as_ref();

        let envelope = ModelEnvelope {
            format_version: FORMAT_VERSION,
            n_trees: self.trees.len(),
            n_features: self.n_features,
            n_classes: self.n_classes,
            feature_names: self.feature_names.clone(),
            forest: self.clone(),
        };

        let bytes = bincode::serialize(&envelope).map_err(|e| RfError::SerializeModel {
            source: e,
        })?;

        std::fs::write(path, &bytes).map_err(|e| RfError::WriteModel {
            path: path.to_path_buf(),
            source: e,
        })?;

        info!(
            size_bytes = bytes.len(),
            n_trees = self.trees.len(),
            "model saved"
        );

        Ok(())
    }

    /// Load a model from a binary file.
    ///
    /// Checks the format version and returns an error on mismatch.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`RfError::ReadModel`] | file read failed |
    /// | [`RfError::DeserializeModel`] | bincode decoding failed |
    /// | [`RfError::IncompatibleModelVersion`] | format version mismatch |
    #[instrument(fields(path = %path.as_ref().display()))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RfError> {
        let path = path.as_ref();

        let bytes = std::fs::read(path).map_err(|e| RfError::ReadModel {
            path: path.to_path_buf(),
            source: e,
        })?;

        let envelope: ModelEnvelope = bincode::deserialize(&bytes).map_err(|e| {
            RfError::DeserializeModel {
                path: path.to_path_buf(),
                source: e,
            }
        })?;

        if envelope.format_version != FORMAT_VERSION {
            return Err(RfError::IncompatibleModelVersion {
                expected: FORMAT_VERSION,
                found: envelope.format_version,
                path: path.to_path_buf(),
            });
        }

        debug!(
            n_trees = envelope.n_trees,
            n_features = envelope.n_features,
            n_classes = envelope.n_classes,
            "model loaded"
        );

        Ok(envelope.forest)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use crate::config::RandomForestConfig;
    use crate::forest::RandomForest;

    fn train_simple_model() -> RandomForest {
        let features = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
            vec![12.0, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let names = vec!["x".to_string(), "y".to_string()];
        let result = RandomForestConfig::new(5)
            .unwrap()
            .with_seed(42)
            .fit(&features, &labels, &names)
            .unwrap();
        result.into_forest()
    }

    #[test]
    fn round_trip_identical_predictions() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("test_model.bin");

        let forest = train_simple_model();

        // Save
        forest.save(&model_path).unwrap();

        // Load
        let loaded = RandomForest::load(&model_path).unwrap();

        // Verify identical predictions
        let test_samples = vec![vec![1.5, 0.0], vec![11.0, 0.0], vec![5.0, 0.0]];
        for sample in &test_samples {
            let orig = forest.predict(sample).unwrap();
            let restored = loaded.predict(sample).unwrap();
            assert_eq!(orig, restored, "predictions differ for sample {sample:?}");

            let orig_proba = forest.predict_proba(sample).unwrap();
            let restored_proba = loaded.predict_proba(sample).unwrap();
            assert_eq!(orig_proba.as_slice(), restored_proba.as_slice());
        }
    }

    #[test]
    fn load_nonexistent_file_error() {
        let err = RandomForest::load("/tmp/nonexistent_model_abc123.bin").unwrap_err();
        assert!(matches!(err, crate::RfError::ReadModel { .. }));
    }

    #[test]
    fn load_corrupt_file_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("corrupt.bin");
        std::fs::write(&path, b"not a valid bincode file").unwrap();
        let err = RandomForest::load(&path).unwrap_err();
        assert!(matches!(err, crate::RfError::DeserializeModel { .. }));
    }
}
