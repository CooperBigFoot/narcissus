//! K-means clustering with DTW distance metric.
//!
//! Provides DTW-based K-means clustering with k-means++ initialization,
//! multi-restart optimization, and elbow-method cluster count selection.

mod config;
mod error;
mod inertia;
mod init;
mod kmeans;
mod label;
mod result;

pub use config::{KMeansConfig, OptimizeConfig};
pub use error::ClusterError;
pub use inertia::Inertia;
pub use label::ClusterLabel;
pub use result::{KMeansResult, KResult, OptimizeResult};
