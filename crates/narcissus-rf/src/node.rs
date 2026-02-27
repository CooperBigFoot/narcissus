use std::fmt;

/// Zero-based feature column index.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct FeatureIndex(usize);

impl FeatureIndex {
    /// Create a new feature index from a zero-based column position.
    #[allow(dead_code)]
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based feature column index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for FeatureIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into a `Vec<Node>` arena, identifying a specific node in a decision tree.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct NodeIndex(usize);

impl NodeIndex {
    /// Create a new node index from a zero-based arena position.
    #[allow(dead_code)]
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based arena index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Criterion-agnostic impurity value (Gini or Entropy).
#[derive(
    Debug, Clone, Copy, PartialEq, PartialOrd,
    serde::Serialize, serde::Deserialize,
)]
pub struct Impurity(f64);

impl Impurity {
    /// Create a new impurity value.
    pub(crate) fn new(value: f64) -> Self {
        Self(value)
    }

    /// Return the raw impurity value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Impurity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

/// A node in a decision tree arena.
///
/// Trees are stored as `Vec<Node>` where children are referenced by
/// [`NodeIndex`] rather than pointers — this is cache-friendly and
/// trivially serializable.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Node {
    /// An interior split node.
    Split {
        /// Feature used for the split.
        feature: FeatureIndex,
        /// Threshold value: samples with feature <= threshold go left.
        threshold: f64,
        /// Index of the left child node.
        left: NodeIndex,
        /// Index of the right child node.
        right: NodeIndex,
        /// Impurity at this node before splitting.
        impurity: Impurity,
        /// Number of training samples that reached this node.
        n_samples: usize,
        /// Weighted decrease in impurity from this split.
        impurity_decrease: f64,
    },
    /// A terminal leaf node.
    Leaf {
        /// Predicted class (argmax of distribution).
        prediction: usize,
        /// Normalized class probability distribution.
        distribution: Vec<f64>,
        /// Impurity at this leaf.
        impurity: Impurity,
        /// Number of training samples in this leaf.
        n_samples: usize,
    },
}

impl Node {
    /// Return the impurity at this node (before splitting for interior nodes).
    #[must_use]
    pub fn impurity(&self) -> Impurity {
        match self {
            Node::Split { impurity, .. } | Node::Leaf { impurity, .. } => *impurity,
        }
    }

    /// Return the number of training samples that reached this node.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        match self {
            Node::Split { n_samples, .. } | Node::Leaf { n_samples, .. } => *n_samples,
        }
    }

    /// Return `true` if this node is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureIndex, Impurity, Node, NodeIndex};

    // --- FeatureIndex ---

    #[test]
    fn feature_index_roundtrip() {
        let fi = FeatureIndex::new(7);
        assert_eq!(fi.index(), 7);
    }

    #[test]
    fn feature_index_display() {
        let fi = FeatureIndex::new(3);
        assert_eq!(format!("{fi}"), "3");
    }

    #[test]
    fn feature_index_ordering() {
        let a = FeatureIndex::new(1);
        let b = FeatureIndex::new(5);
        assert!(a < b);
    }

    #[test]
    fn feature_index_equality() {
        let a = FeatureIndex::new(2);
        let b = FeatureIndex::new(2);
        assert_eq!(a, b);
    }

    // --- NodeIndex ---

    #[test]
    fn node_index_roundtrip() {
        let ni = NodeIndex::new(42);
        assert_eq!(ni.index(), 42);
    }

    #[test]
    fn node_index_display() {
        let ni = NodeIndex::new(0);
        assert_eq!(format!("{ni}"), "0");
    }

    #[test]
    fn node_index_ordering() {
        let a = NodeIndex::new(10);
        let b = NodeIndex::new(20);
        assert!(a < b);
    }

    // --- Impurity ---

    #[test]
    fn impurity_roundtrip() {
        let imp = Impurity::new(0.5);
        assert!((imp.value() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn impurity_display() {
        let imp = Impurity::new(0.333333);
        assert_eq!(format!("{imp}"), "0.333333");
    }

    #[test]
    fn impurity_zero_display() {
        let imp = Impurity::new(0.0);
        assert_eq!(format!("{imp}"), "0.000000");
    }

    #[test]
    fn impurity_ordering() {
        let a = Impurity::new(0.1);
        let b = Impurity::new(0.5);
        assert!(a < b);
    }

    // --- Node ---

    fn make_leaf() -> Node {
        Node::Leaf {
            prediction: 1,
            distribution: vec![0.2, 0.8],
            impurity: Impurity::new(0.32),
            n_samples: 10,
        }
    }

    fn make_split() -> Node {
        Node::Split {
            feature: FeatureIndex::new(2),
            threshold: 3.5,
            left: NodeIndex::new(1),
            right: NodeIndex::new(2),
            impurity: Impurity::new(0.48),
            n_samples: 20,
            impurity_decrease: 0.16,
        }
    }

    #[test]
    fn leaf_is_leaf() {
        assert!(make_leaf().is_leaf());
    }

    #[test]
    fn split_is_not_leaf() {
        assert!(!make_split().is_leaf());
    }

    #[test]
    fn leaf_n_samples() {
        assert_eq!(make_leaf().n_samples(), 10);
    }

    #[test]
    fn split_n_samples() {
        assert_eq!(make_split().n_samples(), 20);
    }

    #[test]
    fn leaf_impurity() {
        assert!((make_leaf().impurity().value() - 0.32).abs() < f64::EPSILON);
    }

    #[test]
    fn split_impurity() {
        assert!((make_split().impurity().value() - 0.48).abs() < f64::EPSILON);
    }
}
