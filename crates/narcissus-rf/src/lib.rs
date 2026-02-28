//! Random Forest classification: train, evaluate, predict.
//!
//! Provides a hand-rolled Random Forest classifier with CART decision trees,
//! Gini/Entropy split criteria, parallel training via rayon, out-of-bag
//! evaluation, feature importance, and model serialization.

mod config;
mod confusion;
mod error;
mod eval;
mod forest;
mod histogram;
mod importance;
mod node;
mod oob;
mod perm_importance;
mod predict;
mod result;
mod serialize;
mod split;
mod tree;

pub use config::{MaxFeatures, OobMode, RandomForestConfig};
pub use confusion::{ClassMetrics, ConfusionMatrix};
pub use error::RfError;
pub use eval::{CrossValidation, CrossValidationResult};
pub use forest::RandomForest;
pub use importance::RankedFeature;
pub use node::{FeatureIndex, Impurity, Node, NodeIndex};
pub use oob::OobScore;
pub use perm_importance::PermutationImportance;
pub use predict::ClassDistribution;
pub use result::{RandomForestResult, TrainingMetadata};
pub use split::{SplitCriterion, SplitMethod};
pub use tree::{DecisionTree, DecisionTreeConfig};
