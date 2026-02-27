//! End-to-end integration tests: CSV -> cluster/optimize -> JSON -> deserialize.

use std::fs;
use std::path::Path;

use narcissus_cluster::{KMeansConfig, OptimizeConfig};
use narcissus_dtw::BandConstraint;
use narcissus_io::{ExperimentName, ResultWriter, TimeSeriesReader};
use tempfile::TempDir;

/// Path to the test fixture directory.
fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn cluster_round_trip() {
    // 1. Read CSV
    let dataset = TimeSeriesReader::new(&fixture_path("valid_6x12.csv"))
        .read()
        .expect("fixture should parse");

    assert_eq!(dataset.basin_ids.len(), 6);
    assert_eq!(dataset.series.len(), 6);

    // 2. Cluster with k=3 (data has 3 natural groups near 0, 5, 10)
    let config = KMeansConfig::new(3, BandConstraint::Unconstrained)
        .unwrap()
        .with_seed(42)
        .with_n_init(5);
    let result = config.fit(&dataset.series).unwrap();

    // 3. Write JSON artifact
    let dir = TempDir::new().unwrap();
    let experiment = ExperimentName::new("cluster_rt".into()).unwrap();
    let writer = ResultWriter::new(dir.path(), experiment).unwrap();
    writer
        .write_cluster(&dataset.basin_ids, &result)
        .unwrap();

    // 4. Deserialize back and verify
    let json_path = dir.path().join("cluster_rt_cluster.json");
    let content: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&json_path).unwrap()).unwrap();

    // Verify experiment name
    assert_eq!(content["experiment"], "cluster_rt");

    // Verify k matches
    assert_eq!(content["k"].as_u64().unwrap(), 3);

    // Verify assignments map has all 6 basin IDs
    let assignments = content["assignments"].as_object().unwrap();
    assert_eq!(assignments.len(), 6);
    for basin_id in &dataset.basin_ids {
        assert!(
            assignments.contains_key(basin_id.as_str()),
            "missing basin {} in assignments",
            basin_id.as_str()
        );
    }

    // Verify all labels are in [0, k)
    for (basin, label) in assignments {
        let l = label.as_u64().unwrap();
        assert!(l < 3, "basin {} has label {} >= k=3", basin, l);
    }

    // Verify centroids array has k=3 entries
    let centroids = content["centroids"].as_array().unwrap();
    assert_eq!(centroids.len(), 3);

    // Verify each centroid has the right number of timesteps (12)
    for (i, centroid) in centroids.iter().enumerate() {
        let values = centroid.as_array().unwrap();
        assert_eq!(
            values.len(),
            12,
            "centroid {} has {} values, expected 12",
            i,
            values.len()
        );
    }

    // Verify cluster_sizes sums to n_basins
    let sizes = content["cluster_sizes"].as_array().unwrap();
    let total: u64 = sizes.iter().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(total, 6);

    // Verify inertia is non-negative
    let inertia = content["inertia"].as_f64().unwrap();
    assert!(inertia >= 0.0, "inertia should be non-negative");

    // Verify basins near the same value got the same label
    // B01 and B02 are near 0, B03 and B04 near 5, B05 and B06 near 10
    let b01_label = assignments["B01"].as_u64().unwrap();
    let b02_label = assignments["B02"].as_u64().unwrap();
    let b03_label = assignments["B03"].as_u64().unwrap();
    let b04_label = assignments["B04"].as_u64().unwrap();
    let b05_label = assignments["B05"].as_u64().unwrap();
    let b06_label = assignments["B06"].as_u64().unwrap();

    assert_eq!(b01_label, b02_label, "B01 and B02 should share a cluster");
    assert_eq!(b03_label, b04_label, "B03 and B04 should share a cluster");
    assert_eq!(b05_label, b06_label, "B05 and B06 should share a cluster");

    // The three groups should have distinct labels
    let mut labels = vec![b01_label, b03_label, b05_label];
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), 3, "should have 3 distinct cluster labels");
}

#[test]
fn optimize_round_trip() {
    // 1. Read CSV
    let dataset = TimeSeriesReader::new(&fixture_path("valid_6x12.csv"))
        .read()
        .expect("fixture should parse");

    // 2. Optimize for k in [2, 4]
    let config = OptimizeConfig::new(2, 4, BandConstraint::Unconstrained)
        .unwrap()
        .with_seed(42)
        .with_n_init(3);
    let result = config.fit(&dataset.series).unwrap();

    // 3. Write JSON artifact
    let dir = TempDir::new().unwrap();
    let experiment = ExperimentName::new("optimize_rt".into()).unwrap();
    let writer = ResultWriter::new(dir.path(), experiment).unwrap();
    writer
        .write_optimize(dataset.series.len(), &result)
        .unwrap();

    // 4. Deserialize back and verify
    let json_path = dir.path().join("optimize_rt_optimize.json");
    let content: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&json_path).unwrap()).unwrap();

    // Verify experiment name
    assert_eq!(content["experiment"], "optimize_rt");

    // Verify n_basins
    assert_eq!(content["n_basins"].as_u64().unwrap(), 6);

    // Verify results array has 3 entries (k=2, 3, 4)
    let results = content["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    // Verify k values are sequential
    let k_values: Vec<u64> = results.iter().map(|r| r["k"].as_u64().unwrap()).collect();
    assert_eq!(k_values, vec![2, 3, 4]);

    // Verify all inertias are positive
    for entry in results {
        let inertia = entry["inertia"].as_f64().unwrap();
        assert!(
            inertia > 0.0,
            "inertia for k={} should be positive",
            entry["k"]
        );
    }

    // Verify inertias are all finite and positive
    let inertias: Vec<f64> = results
        .iter()
        .map(|r| r["inertia"].as_f64().unwrap())
        .collect();
    for (i, &val) in inertias.iter().enumerate() {
        assert!(
            val.is_finite() && val > 0.0,
            "inertia[{}] = {} is invalid",
            i,
            val
        );
    }

    // Verify best_k is present (either a number or null)
    assert!(content.get("best_k").is_some());
}

#[test]
fn reader_fixture_files_match_expected_errors() {
    // empty.csv -> EmptyDataset
    let result = TimeSeriesReader::new(&fixture_path("empty.csv")).read();
    assert!(
        matches!(result, Err(narcissus_io::IoError::EmptyDataset { .. })),
        "empty.csv should give EmptyDataset, got: {:?}",
        result
    );

    // jagged.csv -> InconsistentRowLength
    let result = TimeSeriesReader::new(&fixture_path("jagged.csv")).read();
    assert!(
        matches!(result, Err(narcissus_io::IoError::InconsistentRowLength { .. })),
        "jagged.csv should give InconsistentRowLength, got: {:?}",
        result
    );

    // nan.csv -> NonFiniteValue
    let result = TimeSeriesReader::new(&fixture_path("nan.csv")).read();
    assert!(
        matches!(result, Err(narcissus_io::IoError::NonFiniteValue { .. })),
        "nan.csv should give NonFiniteValue, got: {:?}",
        result
    );

    // inf.csv -> NonFiniteValue
    let result = TimeSeriesReader::new(&fixture_path("inf.csv")).read();
    assert!(
        matches!(result, Err(narcissus_io::IoError::NonFiniteValue { .. })),
        "inf.csv should give NonFiniteValue, got: {:?}",
        result
    );

    // duplicate_ids.csv -> DuplicateBasinId
    let result = TimeSeriesReader::new(&fixture_path("duplicate_ids.csv")).read();
    assert!(
        matches!(result, Err(narcissus_io::IoError::DuplicateBasinId { .. })),
        "duplicate_ids.csv should give DuplicateBasinId, got: {:?}",
        result
    );

    // malformed.csv contains an unclosed quote ("B01,1.0,2.0 with no closing quote).
    // The csv crate (with flexible=true) parses this as a single-column record,
    // which triggers InconsistentRowLength (1 column vs 3 expected in the header).
    let result = TimeSeriesReader::new(&fixture_path("malformed.csv")).read();
    assert!(
        matches!(
            result,
            Err(narcissus_io::IoError::InconsistentRowLength { .. })
        ),
        "malformed.csv should give InconsistentRowLength (unclosed quote parsed as 1-col record), got: {:?}",
        result
    );
}
