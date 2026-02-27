//! DTW distance computation and DBA barycenter averaging.
//!
//! Pure math library — zero I/O. Provides Dynamic Time Warping distance
//! computation with optional Sakoe-Chiba band constraint, warping path
//! extraction, pairwise distance matrices, and DBA barycenter averaging.

mod constraint;
mod dba;
mod distance;
mod dtw;
mod error;
mod matrix;
mod path;
mod series;

pub use constraint::BandConstraint;
pub use dba::{DbaConfig, DbaResult};
pub use distance::DtwDistance;
pub use dtw::Dtw;
pub use error::{DbaError, DtwError};
pub use matrix::DistanceMatrix;
pub use path::{WarpingPath, WarpingStep};
pub use series::{TimeSeries, TimeSeriesView};
