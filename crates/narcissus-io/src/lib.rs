//! File I/O, validation, and serialization for the narcissus pipeline.

mod domain;
mod error;
mod reader;
mod writer;

pub use domain::{BasinId, Dataset, ExperimentName};
pub use error::IoError;
pub use reader::TimeSeriesReader;
pub use writer::ResultWriter;
