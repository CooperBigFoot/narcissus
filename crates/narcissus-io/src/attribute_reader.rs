//! CSV attribute reader with full input validation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{debug, info, instrument};

use crate::domain::{AttributeDataset, BasinId};
use crate::IoError;

/// Reads basin attribute (static feature) data from a CSV file.
///
/// Expected CSV format:
/// - Header row required (first column is basin_id, remaining are feature names)
/// - `basin_id,feature1,feature2,...,featureN`
/// - One row per basin, all rows must have the same number of columns
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`IoError::FileNotFound`] | File doesn't exist or is unreadable |
/// | [`IoError::CsvParse`] | Malformed CSV record |
/// | [`IoError::EmptyDataset`] | Zero data rows after header |
/// | [`IoError::InconsistentRowLength`] | Row has different column count than header |
/// | [`IoError::NonFiniteValue`] | Cell is NaN, Inf, or unparseable float |
/// | [`IoError::DuplicateBasinId`] | Same basin_id appears twice |
/// | [`IoError::NoFeatureColumns`] | Only basin_id column, no feature columns |
pub struct AttributeReader {
    path: PathBuf,
}

impl AttributeReader {
    /// Create a new reader for the given CSV file path.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Read and validate the CSV file, returning an [`AttributeDataset`].
    #[instrument(skip(self), fields(path = %self.path.display()))]
    pub fn read(&self) -> Result<AttributeDataset, IoError> {
        // 1. Open file (FileNotFound on failure)
        let file = std::fs::File::open(&self.path).map_err(|e| IoError::FileNotFound {
            path: self.path.clone(),
            source: e,
        })?;

        // 2. Build CSV reader with headers.
        // flexible(true) allows rows with varying column counts so that our own
        // InconsistentRowLength check fires instead of a low-level CsvParse error.
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(file);

        // 3. Read header to get feature names and expected column count
        let header = rdr.headers().map_err(|e| IoError::CsvParse {
            path: self.path.clone(),
            offset: e.position().map_or(0, |p| p.byte()),
            source: e,
        })?;
        let expected_cols = header.len();
        debug!(expected_cols, "read CSV header");

        // Check for at least one feature column (beyond basin_id)
        if expected_cols < 2 {
            return Err(IoError::NoFeatureColumns {
                path: self.path.clone(),
            });
        }

        // Extract feature names from header (columns 1..n)
        let feature_names: Vec<String> = header.iter().skip(1).map(String::from).collect();

        // 4. Iterate rows with validation
        let mut basin_ids = Vec::new();
        let mut features = Vec::new();
        let mut seen: HashMap<String, usize> = HashMap::new();

        for (row_index, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| IoError::CsvParse {
                path: self.path.clone(),
                offset: e.position().map_or(0, |p| p.byte()),
                source: e,
            })?;

            // Check column count consistency
            if record.len() != expected_cols {
                let basin_id = record.get(0).unwrap_or("").to_string();
                return Err(IoError::InconsistentRowLength {
                    path: self.path.clone(),
                    row_index,
                    basin_id,
                    expected: expected_cols,
                    got: record.len(),
                });
            }

            // Extract basin_id (first column)
            let basin_id_str = record.get(0).unwrap_or("").to_string();

            // Check for duplicate basin IDs
            if let Some(&first_row) = seen.get(&basin_id_str) {
                return Err(IoError::DuplicateBasinId {
                    path: self.path.clone(),
                    basin_id: basin_id_str,
                    first_row,
                    second_row: row_index,
                });
            }
            seen.insert(basin_id_str.clone(), row_index);

            // Parse feature values (columns 1..n)
            let mut row_features = Vec::with_capacity(feature_names.len());
            for col_index in 1..record.len() {
                let raw = record.get(col_index).unwrap_or("");
                let value: f64 = raw.parse().map_err(|_| IoError::NonFiniteValue {
                    path: self.path.clone(),
                    row_index,
                    col_index: col_index - 1,
                    raw: raw.to_string(),
                })?;
                if !value.is_finite() {
                    return Err(IoError::NonFiniteValue {
                        path: self.path.clone(),
                        row_index,
                        col_index: col_index - 1,
                        raw: raw.to_string(),
                    });
                }
                row_features.push(value);
            }

            basin_ids.push(BasinId::new(basin_id_str));
            features.push(row_features);
        }

        // 5. Check for empty dataset
        if basin_ids.is_empty() {
            return Err(IoError::EmptyDataset {
                path: self.path.clone(),
            });
        }

        info!(
            n_basins = basin_ids.len(),
            n_features = feature_names.len(),
            "attribute dataset loaded"
        );

        Ok(AttributeDataset::new(basin_ids, feature_names, features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn read_valid_attributes() {
        let csv = "basin_id,area,slope,elevation\nB01,100.0,0.05,500.0\nB02,200.0,0.10,600.0\nB03,150.0,0.08,550.0\n";
        let f = write_csv(csv);
        let ds = AttributeReader::new(f.path()).read().unwrap();
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 3);
        assert_eq!(ds.feature_names(), &["area", "slope", "elevation"]);
        assert_eq!(ds.basin_ids()[0].as_str(), "B01");
        assert!((ds.features()[0][0] - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_dataset_error() {
        let csv = "basin_id,area,slope\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::EmptyDataset { .. }));
    }

    #[test]
    fn no_feature_columns_error() {
        let csv = "basin_id\nB01\nB02\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::NoFeatureColumns { .. }));
    }

    #[test]
    fn duplicate_basin_id_error() {
        let csv = "basin_id,area\nB01,100.0\nB01,200.0\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::DuplicateBasinId { .. }));
    }

    #[test]
    fn inconsistent_row_length_error() {
        let csv = "basin_id,area,slope\nB01,100.0,0.05\nB02,200.0\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::InconsistentRowLength { .. }));
    }

    #[test]
    fn non_finite_value_error() {
        let csv = "basin_id,area\nB01,NaN\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::NonFiniteValue { .. }));
    }

    #[test]
    fn unparseable_value_error() {
        let csv = "basin_id,area\nB01,abc\n";
        let f = write_csv(csv);
        let err = AttributeReader::new(f.path()).read().unwrap_err();
        assert!(matches!(err, IoError::NonFiniteValue { .. }));
    }
}
