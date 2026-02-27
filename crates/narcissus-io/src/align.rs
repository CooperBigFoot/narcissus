//! Alignment of cluster assignments with basin attributes.

use std::collections::HashMap;

use tracing::{info, instrument, warn};

use crate::domain::{AttributeDataset, BasinId};
use crate::IoError;

/// Aligned dataset: basin attributes matched with cluster labels.
///
/// Basin IDs, features, and labels are stored in parallel vectors —
/// `basin_ids[i]` corresponds to `features[i]` and `labels[i]`.
#[derive(Debug)]
pub struct AlignedData {
    /// Basin IDs in cluster order.
    basin_ids: Vec<BasinId>,
    /// Feature matrix (row-major): `features[sample][feature]`.
    features: Vec<Vec<f64>>,
    /// Cluster labels (as usize) for each basin.
    labels: Vec<usize>,
    /// Feature column names.
    feature_names: Vec<String>,
}

impl AlignedData {
    /// Return the basin IDs.
    #[must_use]
    pub fn basin_ids(&self) -> &[BasinId] {
        &self.basin_ids
    }

    /// Return the feature matrix.
    #[must_use]
    pub fn features(&self) -> &[Vec<f64>] {
        &self.features
    }

    /// Return the cluster labels.
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Return the feature column names.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Return the number of samples.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.basin_ids.len()
    }

    /// Return the number of features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }
}

/// Align cluster assignments with basin attributes.
///
/// Iterates over clustered basins in order and looks up each in the
/// attribute dataset. Extra attribute basins (those not in the cluster
/// set) are silently dropped with a log message. Missing basins cause
/// a hard error.
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`IoError::MissingBasinAttributes`] | A clustered basin is not found in the attribute dataset |
#[instrument(skip_all, fields(n_clustered = cluster_basin_ids.len(), n_attributes = attributes.n_samples()))]
pub fn align(
    cluster_basin_ids: &[BasinId],
    cluster_labels: &[usize],
    attributes: &AttributeDataset,
) -> Result<AlignedData, IoError> {
    // Build lookup: basin_id string -> index in attributes
    let attr_lookup: HashMap<&str, usize> = attributes
        .basin_ids()
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    let mut basin_ids = Vec::with_capacity(cluster_basin_ids.len());
    let mut features = Vec::with_capacity(cluster_basin_ids.len());
    let mut labels = Vec::with_capacity(cluster_basin_ids.len());

    for (i, basin_id) in cluster_basin_ids.iter().enumerate() {
        let attr_idx = attr_lookup.get(basin_id.as_str()).ok_or_else(|| {
            IoError::MissingBasinAttributes {
                basin_id: basin_id.as_str().to_string(),
            }
        })?;

        basin_ids.push(basin_id.clone());
        features.push(attributes.features()[*attr_idx].clone());
        labels.push(cluster_labels[i]);
    }

    let n_dropped = attributes.n_samples() - basin_ids.len();
    if n_dropped > 0 {
        warn!(n_dropped, "dropped extra attribute basins not in cluster set");
    }

    info!(
        n_aligned = basin_ids.len(),
        n_features = attributes.n_features(),
        "alignment complete"
    );

    Ok(AlignedData {
        basin_ids,
        features,
        labels,
        feature_names: attributes.feature_names().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::AttributeDataset;

    fn make_attributes() -> AttributeDataset {
        AttributeDataset::new(
            vec![
                BasinId::new("B01".into()),
                BasinId::new("B02".into()),
                BasinId::new("B03".into()),
                BasinId::new("B04".into()),
            ],
            vec!["area".into(), "slope".into()],
            vec![
                vec![100.0, 0.05],
                vec![200.0, 0.10],
                vec![150.0, 0.08],
                vec![300.0, 0.12],
            ],
        )
    }

    #[test]
    fn valid_alignment() {
        let attrs = make_attributes();
        let cluster_ids = vec![
            BasinId::new("B01".into()),
            BasinId::new("B03".into()),
        ];
        let cluster_labels = vec![0, 1];

        let aligned = align(&cluster_ids, &cluster_labels, &attrs).unwrap();
        assert_eq!(aligned.n_samples(), 2);
        assert_eq!(aligned.labels(), &[0, 1]);
        assert_eq!(aligned.basin_ids()[0].as_str(), "B01");
        assert_eq!(aligned.basin_ids()[1].as_str(), "B03");
        assert!((aligned.features()[0][0] - 100.0).abs() < f64::EPSILON);
        assert!((aligned.features()[1][0] - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn superset_attributes_drops_extras() {
        let attrs = make_attributes();
        // Only cluster B01 and B02, B03 and B04 are extras
        let cluster_ids = vec![
            BasinId::new("B01".into()),
            BasinId::new("B02".into()),
        ];
        let cluster_labels = vec![0, 0];

        let aligned = align(&cluster_ids, &cluster_labels, &attrs).unwrap();
        assert_eq!(aligned.n_samples(), 2);
    }

    #[test]
    fn missing_basin_error() {
        let attrs = make_attributes();
        let cluster_ids = vec![
            BasinId::new("B01".into()),
            BasinId::new("B99".into()), // not in attributes
        ];
        let cluster_labels = vec![0, 1];

        let err = align(&cluster_ids, &cluster_labels, &attrs).unwrap_err();
        assert!(matches!(err, IoError::MissingBasinAttributes { ref basin_id } if basin_id == "B99"));
    }
}
