//! DTW distance computation and DBA barycenter averaging.
//!
//! Pure math library — zero I/O. Provides Dynamic Time Warping distance
//! computation with optional Sakoe-Chiba band constraint, warping path
//! extraction, pairwise distance matrices, and DBA barycenter averaging.

mod constraint;
mod dba;
mod distance;
mod dtw;
mod envelope;
mod error;
mod matrix;
mod path;
mod preprocess;
mod series;
mod ssg;

pub use constraint::BandConstraint;
pub use dba::{DbaConfig, DbaInit, DbaMode, DbaResult};
pub use distance::DtwDistance;
pub use dtw::Dtw;
pub use envelope::{lb_improved, lb_keogh, SeriesEnvelope};
pub use error::{DbaError, DerivativeError, DtwError, PreprocessError};
pub use matrix::DistanceMatrix;
pub use path::{WarpingPath, WarpingStep};
pub use preprocess::{derivative, z_normalize, z_normalize_batch};
pub use series::{TimeSeries, TimeSeriesView};
pub use ssg::{SsgConfig, SsgResult};
