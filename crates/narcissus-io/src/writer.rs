//! JSON result writer for clustering and optimization outputs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use narcissus_cluster::{KMeansResult, OptimizeResult};
use serde::Serialize;
use tracing::{debug, info, instrument};

use crate::domain::{BasinId, ExperimentName};
use crate::IoError;

/// Writes clustering and optimization results to JSON files.
///
/// Creates the output directory on construction if it does not exist.
/// Output files are named `{experiment}_cluster.json` and
/// `{experiment}_optimize.json`.
pub struct ResultWriter {
    output_dir: PathBuf,
    experiment: ExperimentName,
}

impl ResultWriter {
    /// Create a new writer targeting the given directory and experiment name.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::OutputDirCreate`] if the directory cannot be created.
    #[instrument(skip_all, fields(dir = %output_dir.display(), experiment = %experiment))]
    pub fn new(output_dir: &Path, experiment: ExperimentName) -> Result<Self, IoError> {
        fs::create_dir_all(output_dir).map_err(|e| IoError::OutputDirCreate {
            path: output_dir.to_path_buf(),
            source: e,
        })?;
        debug!("output directory ready");
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            experiment,
        })
    }

    /// Write a clustering result to `{experiment}_cluster.json`.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::WriteFile`] if the file cannot be written.
    #[instrument(skip_all)]
    pub fn write_cluster(
        &self,
        basin_ids: &[BasinId],
        result: &KMeansResult,
    ) -> Result<(), IoError> {
        let path = self
            .output_dir
            .join(format!("{}_cluster.json", self.experiment.as_str()));

        // Build assignments map: basin_id → cluster label
        let assignments: BTreeMap<&str, usize> = basin_ids
            .iter()
            .zip(&result.assignments)
            .map(|(id, label)| (id.as_str(), label.index()))
            .collect();

        // Extract centroid values
        let centroids: Vec<Vec<f64>> = result
            .centroids
            .iter()
            .map(|c| c.as_ref().to_vec())
            .collect();

        let artifact = ClusterArtifact {
            experiment: self.experiment.as_str(),
            k: result.centroids.len(),
            inertia: result.inertia.value(),
            converged: result.converged,
            iterations: result.iterations,
            n_init_used: result.n_init_used,
            assignments,
            centroids,
            cluster_sizes: result.cluster_sizes(),
        };

        let json = serde_json::to_string_pretty(&artifact).expect("serialization cannot fail");
        fs::write(&path, &json).map_err(|e| IoError::WriteFile {
            path: path.clone(),
            source: e,
        })?;

        info!(path = %path.display(), "cluster result written");
        Ok(())
    }

    /// Write an optimization result to `{experiment}_optimize.json`.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::WriteFile`] if the file cannot be written.
    #[instrument(skip_all)]
    pub fn write_optimize(
        &self,
        n_basins: usize,
        result: &OptimizeResult,
    ) -> Result<(), IoError> {
        let path = self
            .output_dir
            .join(format!("{}_optimize.json", self.experiment.as_str()));

        let results: Vec<KResultEntry> = result
            .results
            .iter()
            .map(|r| KResultEntry {
                k: r.k,
                inertia: r.inertia.value(),
            })
            .collect();

        let artifact = OptimizeArtifact {
            experiment: self.experiment.as_str(),
            n_basins,
            best_k: result.best_k(),
            results,
        };

        let json = serde_json::to_string_pretty(&artifact).expect("serialization cannot fail");
        fs::write(&path, &json).map_err(|e| IoError::WriteFile {
            path: path.clone(),
            source: e,
        })?;

        info!(path = %path.display(), "optimize result written");
        Ok(())
    }

    /// Write evaluation results to `{experiment}_evaluate.json`.
    ///
    /// Uses shadow structs to accept primitives — the writer has no
    /// dependency on `narcissus-rf`.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::WriteFile`] if the file cannot be written.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    pub fn write_evaluation(
        &self,
        cv_accuracy_mean: f64,
        cv_accuracy_std: f64,
        fold_accuracies: &[f64],
        oob_accuracy: Option<f64>,
        feature_names: &[String],
        feature_importances: &[f64],
        feature_ranks: &[usize],
        confusion_matrix: &[Vec<usize>],
        n_classes: usize,
        class_metrics: &[(f64, f64, f64, usize)], // (precision, recall, f1, support)
    ) -> Result<(), IoError> {
        let path = self
            .output_dir
            .join(format!("{}_evaluate.json", self.experiment.as_str()));

        let features: Vec<FeatureEntry> = feature_names
            .iter()
            .zip(feature_importances.iter())
            .zip(feature_ranks.iter())
            .map(|((name, &importance), &rank)| FeatureEntry {
                name: name.as_str(),
                importance,
                rank,
            })
            .collect();

        let classes: Vec<ClassEntry> = class_metrics
            .iter()
            .enumerate()
            .map(|(i, &(precision, recall, f1, support))| ClassEntry {
                class: i,
                precision,
                recall,
                f1,
                support,
            })
            .collect();

        let artifact = EvaluateArtifact {
            experiment: self.experiment.as_str(),
            cv_accuracy_mean,
            cv_accuracy_std,
            fold_accuracies,
            oob_accuracy,
            feature_importances: features,
            confusion_matrix,
            n_classes,
            class_metrics: classes,
        };

        let json = serde_json::to_string_pretty(&artifact).expect("serialization cannot fail");
        fs::write(&path, &json).map_err(|e| IoError::WriteFile {
            path: path.clone(),
            source: e,
        })?;

        info!(path = %path.display(), "evaluation result written");
        Ok(())
    }

    /// Write predictions to `{experiment}_predict.json`.
    ///
    /// Each entry is a `(basin_id, Vec<(class, probability)>)` pair.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::WriteFile`] if the file cannot be written.
    #[instrument(skip_all)]
    pub fn write_predictions(
        &self,
        predictions: &[(String, Vec<(usize, f64)>)],
    ) -> Result<(), IoError> {
        let path = self
            .output_dir
            .join(format!("{}_predict.json", self.experiment.as_str()));

        let entries: Vec<PredictionEntry> = predictions
            .iter()
            .map(|(basin_id, top_k)| {
                let classes: Vec<PredictionClass> = top_k
                    .iter()
                    .map(|&(class, probability)| PredictionClass { class, probability })
                    .collect();
                PredictionEntry {
                    basin_id: basin_id.as_str(),
                    predicted_class: top_k.first().map(|&(c, _)| c).unwrap_or(0),
                    top_k: classes,
                }
            })
            .collect();

        let artifact = PredictArtifact {
            experiment: self.experiment.as_str(),
            n_basins: predictions.len(),
            predictions: entries,
        };

        let json = serde_json::to_string_pretty(&artifact).expect("serialization cannot fail");
        fs::write(&path, &json).map_err(|e| IoError::WriteFile {
            path: path.clone(),
            source: e,
        })?;

        info!(path = %path.display(), "predictions written");
        Ok(())
    }

    /// Return the path where the model binary should be saved.
    ///
    /// Does not write anything — just computes `{output_dir}/{experiment}_model.bin`.
    #[must_use]
    pub fn model_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}_model.bin", self.experiment.as_str()))
    }
}

// --- Shadow structs for JSON serialization ---

#[derive(Serialize)]
struct ClusterArtifact<'a> {
    experiment: &'a str,
    k: usize,
    inertia: f64,
    converged: bool,
    iterations: usize,
    n_init_used: usize,
    assignments: BTreeMap<&'a str, usize>,
    centroids: Vec<Vec<f64>>,
    cluster_sizes: Vec<usize>,
}

#[derive(Serialize)]
struct OptimizeArtifact<'a> {
    experiment: &'a str,
    n_basins: usize,
    best_k: Option<usize>,
    results: Vec<KResultEntry>,
}

#[derive(Serialize)]
struct KResultEntry {
    k: usize,
    inertia: f64,
}

#[derive(Serialize)]
struct EvaluateArtifact<'a> {
    experiment: &'a str,
    cv_accuracy_mean: f64,
    cv_accuracy_std: f64,
    fold_accuracies: &'a [f64],
    oob_accuracy: Option<f64>,
    feature_importances: Vec<FeatureEntry<'a>>,
    confusion_matrix: &'a [Vec<usize>],
    n_classes: usize,
    class_metrics: Vec<ClassEntry>,
}

#[derive(Serialize)]
struct FeatureEntry<'a> {
    name: &'a str,
    importance: f64,
    rank: usize,
}

#[derive(Serialize)]
struct ClassEntry {
    class: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    support: usize,
}

#[derive(Serialize)]
struct PredictArtifact<'a> {
    experiment: &'a str,
    n_basins: usize,
    predictions: Vec<PredictionEntry<'a>>,
}

#[derive(Serialize)]
struct PredictionEntry<'a> {
    basin_id: &'a str,
    predicted_class: usize,
    top_k: Vec<PredictionClass>,
}

#[derive(Serialize)]
struct PredictionClass {
    class: usize,
    probability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use narcissus_cluster::{KMeansConfig, OptimizeConfig};
    use narcissus_dtw::{BandConstraint, TimeSeries};
    use tempfile::TempDir;

    fn test_series() -> Vec<TimeSeries> {
        vec![
            TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![0.1, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![5.0, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![5.1, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0]).unwrap(),
            TimeSeries::new(vec![10.1, 10.0, 10.0, 10.0]).unwrap(),
        ]
    }

    fn test_basin_ids() -> Vec<BasinId> {
        vec![
            BasinId::new("B01".into()),
            BasinId::new("B02".into()),
            BasinId::new("B03".into()),
            BasinId::new("B04".into()),
            BasinId::new("B05".into()),
            BasinId::new("B06".into()),
        ]
    }

    #[test]
    fn write_cluster_json_structure() {
        let dir = TempDir::new().unwrap();
        let experiment = ExperimentName::new("test_run".into()).unwrap();
        let writer = ResultWriter::new(dir.path(), experiment).unwrap();

        let series = test_series();
        let config = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_seed(42)
            .with_n_init(1);
        let result = config.fit(&series).unwrap();

        let basin_ids = test_basin_ids();
        writer.write_cluster(&basin_ids, &result).unwrap();

        let path = dir.path().join("test_run_cluster.json");
        assert!(path.exists());

        let content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(content["experiment"], "test_run");
        assert_eq!(content["k"], 3);
        assert!(content["inertia"].is_number());
        assert!(content["converged"].is_boolean());
        assert!(content["iterations"].is_number());
        assert!(content["assignments"].is_object());
        assert!(content["centroids"].is_array());
        assert_eq!(content["centroids"].as_array().unwrap().len(), 3);
        assert!(content["cluster_sizes"].is_array());
    }

    #[test]
    fn write_cluster_assignments_map_correctness() {
        let dir = TempDir::new().unwrap();
        let experiment = ExperimentName::new("assign_test".into()).unwrap();
        let writer = ResultWriter::new(dir.path(), experiment).unwrap();

        let series = test_series();
        let config = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_seed(42)
            .with_n_init(1);
        let result = config.fit(&series).unwrap();

        let basin_ids = test_basin_ids();
        writer.write_cluster(&basin_ids, &result).unwrap();

        let path = dir.path().join("assign_test_cluster.json");
        let content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        let assignments = content["assignments"].as_object().unwrap();
        // All 6 basins should be in the assignments map
        assert_eq!(assignments.len(), 6);
        assert!(assignments.contains_key("B01"));
        assert!(assignments.contains_key("B06"));
        // All labels should be in range [0, k)
        for (_basin, label) in assignments {
            let l = label.as_u64().unwrap() as usize;
            assert!(l < 3, "cluster label {} out of range", l);
        }
    }

    #[test]
    fn write_optimize_json_structure() {
        let dir = TempDir::new().unwrap();
        let experiment = ExperimentName::new("opt_test".into()).unwrap();
        let writer = ResultWriter::new(dir.path(), experiment).unwrap();

        let series = test_series();
        let config = OptimizeConfig::new(2, 4, BandConstraint::Unconstrained)
            .unwrap()
            .with_seed(42)
            .with_n_init(1);
        let result = config.fit(&series).unwrap();

        writer.write_optimize(series.len(), &result).unwrap();

        let path = dir.path().join("opt_test_optimize.json");
        assert!(path.exists());

        let content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(content["experiment"], "opt_test");
        assert_eq!(content["n_basins"], 6);
        assert!(content["results"].is_array());
        let results = content["results"].as_array().unwrap();
        assert_eq!(results.len(), 3); // k=2,3,4
        // Each entry has k and inertia
        for entry in results {
            assert!(entry["k"].is_number());
            assert!(entry["inertia"].is_number());
        }
    }

    #[test]
    fn write_cluster_creates_output_dir() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested").join("deep");
        let experiment = ExperimentName::new("nested_test".into()).unwrap();
        let writer = ResultWriter::new(&nested, experiment).unwrap();

        let series = test_series();
        let config = KMeansConfig::new(2, BandConstraint::Unconstrained)
            .unwrap()
            .with_seed(42)
            .with_n_init(1);
        let result = config.fit(&series).unwrap();

        let basin_ids = test_basin_ids();
        writer.write_cluster(&basin_ids, &result).unwrap();

        assert!(nested.join("nested_test_cluster.json").exists());
    }

    #[test]
    fn invalid_experiment_name_rejected() {
        let result = ExperimentName::new("bad name!".into());
        assert!(matches!(result, Err(IoError::InvalidExperimentName { .. })));
    }

    #[test]
    fn write_optimize_best_k_present() {
        let dir = TempDir::new().unwrap();
        let experiment = ExperimentName::new("bestk_test".into()).unwrap();
        let writer = ResultWriter::new(dir.path(), experiment).unwrap();

        let series = test_series();
        let config = OptimizeConfig::new(2, 4, BandConstraint::Unconstrained)
            .unwrap()
            .with_seed(42)
            .with_n_init(1);
        let result = config.fit(&series).unwrap();

        writer.write_optimize(series.len(), &result).unwrap();

        let path = dir.path().join("bestk_test_optimize.json");
        let content: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        // best_k should be present (either a number or null)
        assert!(content.get("best_k").is_some());
    }
}
