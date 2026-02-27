//! File I/O, validation, and serialization for the narcissus pipeline.

mod align;
mod attribute_reader;
mod domain;
mod error;
mod reader;
mod writer;

pub use align::{align, AlignedData};
pub use attribute_reader::AttributeReader;
pub use domain::{AttributeDataset, BasinId, Dataset, ExperimentName};
pub use error::IoError;
pub use reader::TimeSeriesReader;
pub use writer::ResultWriter;
