//! CSV time series reader with full input validation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use narcissus_dtw::TimeSeries;
use tracing::{debug, info, instrument};

use crate::domain::{BasinId, Dataset};
use crate::IoError;

/// Reads time series data from a CSV file.
///
/// Expected CSV format:
/// - Header row required (first column is basin_id, remaining are positional time steps)
/// - `basin_id,t0,t1,...,tn`
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
pub struct TimeSeriesReader {
    path: PathBuf,
}

impl TimeSeriesReader {
    /// Create a new reader for the given CSV file path.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Read and validate the CSV file, returning a [`Dataset`].
    #[instrument(skip(self), fields(path = %self.path.display()))]
    pub fn read(&self) -> Result<Dataset, IoError> {
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

        // 3. Read header to determine expected column count
        let header = rdr.headers().map_err(|e| IoError::CsvParse {
            path: self.path.clone(),
            offset: e.position().map_or(0, |p| p.byte()),
            source: e,
        })?;
        let expected_cols = header.len();
        debug!(expected_cols, "read CSV header");

        // 4. Iterate rows with validation
        let mut basin_ids = Vec::new();
        let mut series = Vec::new();
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

            // Parse time step values (columns 1..n)
            let mut values = Vec::with_capacity(expected_cols - 1);
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
                values.push(value);
            }

            // Build TimeSeries (this also validates non-empty and finite)
            // We already validated, so this should not fail, but handle gracefully
            let ts = TimeSeries::new(values).map_err(|_| IoError::EmptyDataset {
                path: self.path.clone(),
            })?;

            basin_ids.push(BasinId::new(basin_id_str));
            series.push(ts);
        }

        // 5. Check for empty dataset
        if basin_ids.is_empty() {
            return Err(IoError::EmptyDataset {
                path: self.path.clone(),
            });
        }

        info!(
            n_basins = basin_ids.len(),
            n_timesteps = series.first().map_or(0, |s| s.len()),
            "dataset loaded"
        );

        Ok(Dataset { basin_ids, series })
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
    fn read_valid_6_basins() {
        let csv = "basin_id,t0,t1,t2,t3\nB01,0.0,0.1,0.0,0.1\nB02,0.1,0.0,0.1,0.0\nB03,5.0,5.1,5.0,5.1\nB04,5.1,5.0,5.1,5.0\nB05,10.0,10.1,10.0,10.1\nB06,10.1,10.0,10.1,10.0\n";
        let f = write_csv(csv);
        let ds = TimeSeriesReader::new(f.path()).read().unwrap();
        assert_eq!(ds.basin_ids.len(), 6);
        assert_eq!(ds.series.len(), 6);
        assert_eq!(ds.basin_ids[0].as_str(), "B01");
        assert_eq!(ds.basin_ids[5].as_str(), "B06");
    }

    #[test]
    fn read_valid_1_basin() {
        let csv = "basin_id,t0,t1,t2,t3\nONLY,1.0,2.0,3.0,4.0\n";
        let f = write_csv(csv);
        let ds = TimeSeriesReader::new(f.path()).read().unwrap();
        assert_eq!(ds.basin_ids.len(), 1);
        assert_eq!(ds.series[0].as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn value_round_trip() {
        let csv = "basin_id,t0,t1\nA,1.23456789,9.87654321\n";
        let f = write_csv(csv);
        let ds = TimeSeriesReader::new(f.path()).read().unwrap();
        let vals = ds.series[0].as_ref();
        assert!((vals[0] - 1.23456789).abs() < 1e-12);
        assert!((vals[1] - 9.87654321).abs() < 1e-12);
    }

    #[test]
    fn insertion_order_preserved() {
        let csv = "basin_id,t0\nZZZ,1.0\nAAA,2.0\nMMM,3.0\n";
        let f = write_csv(csv);
        let ds = TimeSeriesReader::new(f.path()).read().unwrap();
        assert_eq!(ds.basin_ids[0].as_str(), "ZZZ");
        assert_eq!(ds.basin_ids[1].as_str(), "AAA");
        assert_eq!(ds.basin_ids[2].as_str(), "MMM");
    }

    #[test]
    fn error_file_not_found() {
        let result = TimeSeriesReader::new(Path::new("/nonexistent/file.csv")).read();
        assert!(matches!(result, Err(IoError::FileNotFound { .. })));
    }

    #[test]
    fn error_empty_dataset() {
        let csv = "basin_id,t0,t1,t2\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(result, Err(IoError::EmptyDataset { .. })));
    }

    #[test]
    fn error_inconsistent_row_length() {
        let csv = "basin_id,t0,t1,t2\nB01,1.0,2.0,3.0\nB02,1.0,2.0\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(
            result,
            Err(IoError::InconsistentRowLength { row_index: 1, .. })
        ));
    }

    #[test]
    fn error_non_finite_nan() {
        let csv = "basin_id,t0,t1\nB01,1.0,NaN\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(result, Err(IoError::NonFiniteValue { .. })));
    }

    #[test]
    fn error_non_finite_inf() {
        let csv = "basin_id,t0,t1\nB01,1.0,Inf\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(result, Err(IoError::NonFiniteValue { .. })));
    }

    #[test]
    fn error_unparseable_value() {
        let csv = "basin_id,t0,t1\nB01,1.0,abc\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(result, Err(IoError::NonFiniteValue { .. })));
    }

    #[test]
    fn error_duplicate_basin_id() {
        let csv = "basin_id,t0,t1\nB01,1.0,2.0\nB02,3.0,4.0\nB01,5.0,6.0\n";
        let f = write_csv(csv);
        let result = TimeSeriesReader::new(f.path()).read();
        assert!(matches!(
            result,
            Err(IoError::DuplicateBasinId {
                first_row: 0,
                second_row: 2,
                ..
            })
        ));
    }
}
