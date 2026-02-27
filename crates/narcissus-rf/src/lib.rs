//! Random Forest classification: train, evaluate, predict.
//!
//! Provides a hand-rolled Random Forest classifier with CART decision trees,
//! Gini/Entropy split criteria, parallel training via rayon, out-of-bag
//! evaluation, feature importance, and model serialization.

mod confusion;
mod error;
mod node;
mod split;
mod tree;

pub use confusion::{ClassMetrics, ConfusionMatrix};
pub use error::RfError;
pub use node::{FeatureIndex, Impurity, Node, NodeIndex};
pub use split::SplitCriterion;
pub use tree::{DecisionTree, DecisionTreeConfig};
