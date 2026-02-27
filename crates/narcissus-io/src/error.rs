//! I/O error types for narcissus-io.

use std::path::PathBuf;

/// Errors from file I/O, CSV parsing, and result serialization.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Returned when the input file does not exist or is unreadable.
    #[error("file not found: {path}")]
    FileNotFound {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Returned when the CSV parser encounters a malformed record.
    #[error("CSV parse error in {path} at byte offset {offset}")]
    CsvParse {
        /// Path to the CSV file.
        path: PathBuf,
        /// Byte offset where the error occurred.
        offset: u64,
        /// Underlying CSV error.
        source: csv::Error,
    },

    /// Returned when the CSV file contains a header but zero data rows.
    #[error("empty dataset (no data rows) in {path}")]
    EmptyDataset {
        /// Path to the CSV file.
        path: PathBuf,
    },

    /// Returned when a data row has a different number of columns than the header.
    #[error("inconsistent row length in {path}: row {row_index} (basin {basin_id}) has {got} columns, expected {expected}")]
    InconsistentRowLength {
        /// Path to the CSV file.
        path: PathBuf,
        /// Zero-based row index (excluding header).
        row_index: usize,
        /// Basin ID of the offending row.
        basin_id: String,
        /// Expected number of columns (from header).
        expected: usize,
        /// Actual number of columns in this row.
        got: usize,
    },

    /// Returned when a cell value is NaN, Inf, or otherwise not a finite float.
    #[error("non-finite value in {path}: row {row_index}, column {col_index}, raw value \"{raw}\"")]
    NonFiniteValue {
        /// Path to the CSV file.
        path: PathBuf,
        /// Zero-based row index (excluding header).
        row_index: usize,
        /// Zero-based column index (excluding basin_id column).
        col_index: usize,
        /// The raw string value that failed to parse.
        raw: String,
    },

    /// Returned when the same basin ID appears more than once.
    #[error("duplicate basin ID \"{basin_id}\" in {path}: first at row {first_row}, again at row {second_row}")]
    DuplicateBasinId {
        /// Path to the CSV file.
        path: PathBuf,
        /// The duplicated basin ID.
        basin_id: String,
        /// Zero-based row index of the first occurrence.
        first_row: usize,
        /// Zero-based row index of the second occurrence.
        second_row: usize,
    },

    /// Returned when the experiment name contains characters outside `[a-zA-Z0-9_-]`.
    #[error("invalid experiment name \"{name}\": must match [a-zA-Z0-9_-]+")]
    InvalidExperimentName {
        /// The invalid name.
        name: String,
    },

    /// Returned when the output directory cannot be created.
    #[error("cannot create output directory {path}")]
    OutputDirCreate {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Returned when a result file cannot be written.
    #[error("cannot write file {path}")]
    WriteFile {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}
