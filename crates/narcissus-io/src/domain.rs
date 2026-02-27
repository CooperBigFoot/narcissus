//! Domain types for narcissus-io.

use narcissus_dtw::TimeSeries;

use crate::IoError;

/// A hydrological basin identifier.
///
/// Wraps a non-empty string parsed from the first column of the input CSV.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BasinId(String);

impl BasinId {
    /// Create a new basin ID from a non-empty string.
    pub(crate) fn new(id: String) -> Self {
        debug_assert!(!id.is_empty(), "basin ID must not be empty");
        Self(id)
    }

    /// Return the basin ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BasinId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A validated experiment name for output file naming.
///
/// Must match `[a-zA-Z0-9_-]+`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentName(String);

impl ExperimentName {
    /// Parse and validate an experiment name.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidExperimentName`] if the name is empty or
    /// contains characters outside `[a-zA-Z0-9_-]`.
    pub fn new(name: String) -> Result<Self, IoError> {
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(IoError::InvalidExperimentName { name });
        }
        Ok(Self(name))
    }

    /// Return the experiment name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ExperimentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A dataset of basin attributes (static features) for classification.
///
/// Produced by [`AttributeReader`](crate::AttributeReader). Basin IDs,
/// feature names, and feature rows are stored in parallel vectors —
/// `basin_ids[i]` corresponds to `features[i]`.
#[derive(Debug)]
pub struct AttributeDataset {
    /// Basin identifiers in insertion order.
    basin_ids: Vec<BasinId>,
    /// Feature column names from the CSV header.
    feature_names: Vec<String>,
    /// Feature values: `features[sample_index][feature_index]`.
    features: Vec<Vec<f64>>,
}

impl AttributeDataset {
    /// Create a new attribute dataset.
    pub(crate) fn new(
        basin_ids: Vec<BasinId>,
        feature_names: Vec<String>,
        features: Vec<Vec<f64>>,
    ) -> Self {
        Self { basin_ids, feature_names, features }
    }

    /// Return the basin IDs.
    #[must_use]
    pub fn basin_ids(&self) -> &[BasinId] {
        &self.basin_ids
    }

    /// Return the feature column names.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Return the feature matrix (row-major).
    #[must_use]
    pub fn features(&self) -> &[Vec<f64>] {
        &self.features
    }

    /// Return the number of samples (basins).
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.basin_ids.len()
    }

    /// Return the number of feature columns.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }
}

/// A dataset of time series with associated basin identifiers.
///
/// Produced by [`TimeSeriesReader`](crate::TimeSeriesReader). Basin IDs and
/// series are stored in parallel vectors — `basin_ids[i]` corresponds to
/// `series[i]`.
#[derive(Debug)]
pub struct Dataset {
    /// Basin identifiers in insertion order (matching the CSV row order).
    pub basin_ids: Vec<BasinId>,
    /// Validated time series in the same order as `basin_ids`.
    pub series: Vec<TimeSeries>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basin_id_as_str_returns_inner() {
        let id = BasinId::new("USGS_01013500".to_string());
        assert_eq!(id.as_str(), "USGS_01013500");
    }

    #[test]
    fn experiment_name_valid() {
        let name = ExperimentName::new("my-experiment_01".to_string());
        assert!(name.is_ok());
        assert_eq!(name.unwrap().as_str(), "my-experiment_01");
    }

    #[test]
    fn experiment_name_rejects_empty() {
        let name = ExperimentName::new(String::new());
        assert!(matches!(name, Err(IoError::InvalidExperimentName { .. })));
    }

    #[test]
    fn experiment_name_rejects_special_chars() {
        let name = ExperimentName::new("my experiment!".to_string());
        assert!(matches!(name, Err(IoError::InvalidExperimentName { .. })));
    }
}
