use std::path::PathBuf;

/// Errors from Random Forest operations.
#[derive(Debug, thiserror::Error)]
pub enum RfError {
    /// Returned when n_trees is zero.
    #[error("n_trees must be at least 1, got {n_trees}")]
    InvalidTreeCount {
        /// The invalid n_trees value provided.
        n_trees: usize,
    },

    /// Returned when max_depth is zero.
    #[error("max_depth must be at least 1, got {max_depth}")]
    InvalidMaxDepth {
        /// The invalid max_depth value provided.
        max_depth: usize,
    },

    /// Returned when min_samples_split is less than 2.
    #[error("min_samples_split must be at least 2, got {min_samples_split}")]
    InvalidMinSamplesSplit {
        /// The invalid min_samples_split value provided.
        min_samples_split: usize,
    },

    /// Returned when min_samples_leaf is zero.
    #[error("min_samples_leaf must be at least 1, got {min_samples_leaf}")]
    InvalidMinSamplesLeaf {
        /// The invalid min_samples_leaf value provided.
        min_samples_leaf: usize,
    },

    /// Returned when max_features resolves to 0 or exceeds n_features.
    #[error("max_features resolved to {max_features}, but must be in [1, {n_features}]")]
    InvalidMaxFeatures {
        /// The resolved max_features value.
        max_features: usize,
        /// The number of features in the dataset.
        n_features: usize,
    },

    /// Returned when bootstrap_fraction is not in (0.0, 1.0].
    #[error("bootstrap_fraction must be in (0.0, 1.0], got {fraction}")]
    InvalidBootstrapFraction {
        /// The invalid bootstrap_fraction value provided.
        fraction: f64,
    },

    /// Returned when n_folds is less than 2.
    #[error("n_folds must be at least 2, got {n_folds}")]
    InvalidFoldCount {
        /// The invalid n_folds value provided.
        n_folds: usize,
    },

    /// Returned when the training dataset has zero samples.
    #[error("training dataset has zero samples")]
    EmptyDataset,

    /// Returned when the training dataset has zero feature columns.
    #[error("training dataset has zero feature columns")]
    ZeroFeatures,

    /// Returned when a sample has a different number of features than expected.
    #[error("sample {sample_index} has {got} features, expected {expected}")]
    FeatureCountMismatch {
        /// The expected number of features.
        expected: usize,
        /// The actual number of features in the sample.
        got: usize,
        /// The zero-based index of the offending sample.
        sample_index: usize,
    },

    /// Returned when a sample has a different number of features at prediction time.
    #[error("prediction input has {got} features, expected {expected}")]
    PredictionFeatureMismatch {
        /// The expected number of features.
        expected: usize,
        /// The actual number of features in the prediction input.
        got: usize,
    },

    /// Returned when a training value is NaN or infinite.
    #[error("non-finite value at sample {sample_index}, feature {feature_index}")]
    NonFiniteValue {
        /// The zero-based index of the offending sample.
        sample_index: usize,
        /// The zero-based index of the offending feature column.
        feature_index: usize,
    },

    /// Returned when a class has fewer samples than the number of folds.
    #[error("class {class} has only {count} samples, need at least {n_folds} for stratified CV")]
    TooFewSamplesForFolds {
        /// The class label with insufficient samples.
        class: usize,
        /// The number of samples belonging to that class.
        count: usize,
        /// The requested number of folds.
        n_folds: usize,
    },

    /// Returned when OOB evaluation fails (no sample has any OOB tree).
    #[error("OOB evaluation failed: {reason}")]
    OobEvaluationFailed {
        /// Human-readable description of why OOB evaluation failed.
        reason: String,
    },

    /// Returned when model serialization fails.
    #[error("failed to serialize model")]
    SerializeModel {
        /// The underlying bincode error.
        source: Box<bincode::ErrorKind>,
    },

    /// Returned when model deserialization fails.
    #[error("failed to deserialize model from {path}")]
    DeserializeModel {
        /// Path to the model file that could not be deserialized.
        path: PathBuf,
        /// The underlying bincode error.
        source: Box<bincode::ErrorKind>,
    },

    /// Returned when writing the model file fails.
    #[error("failed to write model to {path}")]
    WriteModel {
        /// Path to the file that could not be written.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Returned when reading the model file fails.
    #[error("failed to read model from {path}")]
    ReadModel {
        /// Path to the file that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Returned when loading a model with an incompatible format version.
    #[error("incompatible model version in {path}: expected {expected}, found {found}")]
    IncompatibleModelVersion {
        /// The model format version this build expects.
        expected: u32,
        /// The model format version found in the file.
        found: u32,
        /// Path to the model file with the incompatible version.
        path: PathBuf,
    },
}
