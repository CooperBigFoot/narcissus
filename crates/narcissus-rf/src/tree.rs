use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tracing::{debug, instrument};

use crate::{
    RfError,
    histogram::FeatureBins,
    node::{Node, NodeIndex},
    split::{SplitCriterion, SplitMethod, find_split_with_bins},
};

/// Configuration for a single CART decision tree.
///
/// Construct via [`DecisionTreeConfig::new`], then chain `with_*` methods.
///
/// # Defaults
///
/// | Parameter           | Default             |
/// |---------------------|---------------------|
/// | `criterion`         | `Gini`              |
/// | `split_method`      | `Exact`             |
/// | `max_depth`         | `None` (unlimited)  |
/// | `min_samples_split` | 2                   |
/// | `min_samples_leaf`  | 1                   |
/// | `max_features`      | `None` (all features) |
/// | `seed`              | 42                  |
#[derive(Debug, Clone)]
pub struct DecisionTreeConfig {
    pub(crate) criterion: SplitCriterion,
    pub(crate) split_method: SplitMethod,
    pub(crate) max_depth: Option<usize>,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) max_features: Option<usize>,
    pub(crate) seed: u64,
}

impl DecisionTreeConfig {
    /// Create a new config with default values.
    ///
    /// All parameters use the defaults shown in the struct-level documentation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            split_method: SplitMethod::Exact,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            seed: 42,
        }
    }

    /// Set the split quality criterion.
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the split-finding strategy.
    #[must_use]
    pub fn with_split_method(mut self, split_method: SplitMethod) -> Self {
        self.split_method = split_method;
        self
    }

    /// Set the maximum tree depth.
    ///
    /// `None` means grow until all leaves are pure or stopping conditions
    /// are met. `Some(d)` limits depth to `d` levels (root is depth 0).
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to attempt a split.
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required in each leaf after a split.
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the maximum number of features to consider at each split.
    ///
    /// `None` means consider all features.
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    // --- Getters ---

    /// Return the split criterion.
    #[must_use]
    pub fn criterion(&self) -> SplitCriterion {
        self.criterion
    }

    /// Return the split-finding strategy.
    #[must_use]
    pub fn split_method(&self) -> SplitMethod {
        self.split_method
    }

    /// Return the maximum depth limit, if any.
    #[must_use]
    pub fn max_depth(&self) -> Option<usize> {
        self.max_depth
    }

    /// Return the minimum samples required to split a node.
    #[must_use]
    pub fn min_samples_split(&self) -> usize {
        self.min_samples_split
    }

    /// Return the minimum samples required in each leaf.
    #[must_use]
    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    /// Return the maximum features to consider per split, if set.
    #[must_use]
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// Return the random seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Train a decision tree on the provided row-major dataset.
    ///
    /// `features[sample_idx][feature_idx]` — row-major layout.
    /// `labels[sample_idx]` — class labels (zero-based).
    ///
    /// # Errors
    ///
    /// | Variant                  | When                                                        |
    /// |--------------------------|-------------------------------------------------------------|
    /// | [`RfError::EmptyDataset`]          | `features` is empty                               |
    /// | [`RfError::ZeroFeatures`]          | rows have zero feature columns                    |
    /// | [`RfError::FeatureCountMismatch`]  | rows have inconsistent lengths                    |
    /// | [`RfError::NonFiniteValue`]        | any value is NaN or infinite                      |
    /// | [`RfError::InvalidMaxFeatures`]    | `max_features` resolves outside [1, n_features]   |
    /// | [`RfError::InvalidMaxDepth`]       | `max_depth` is `Some(0)`                          |
    /// | [`RfError::InvalidMinSamplesSplit`]| `min_samples_split` < 2                           |
    /// | [`RfError::InvalidMinSamplesLeaf`] | `min_samples_leaf` < 1                            |
    #[instrument(skip(self, features, labels), fields(n_samples = features.len()))]
    pub fn fit(&self, features: &[Vec<f64>], labels: &[usize]) -> Result<DecisionTree, RfError> {
        // --- Validate inputs ---
        if features.is_empty() {
            return Err(RfError::EmptyDataset);
        }

        let n_samples = features.len();
        let n_features = features[0].len();

        if n_features == 0 {
            return Err(RfError::ZeroFeatures);
        }

        for (sample_index, row) in features.iter().enumerate() {
            if row.len() != n_features {
                return Err(RfError::FeatureCountMismatch {
                    expected: n_features,
                    got: row.len(),
                    sample_index,
                });
            }
            for (feature_index, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    return Err(RfError::NonFiniteValue {
                        sample_index,
                        feature_index,
                    });
                }
            }
        }

        // --- Validate config ---
        if let Some(d) = self.max_depth
            && d == 0
        {
            return Err(RfError::InvalidMaxDepth { max_depth: 0 });
        }

        if self.min_samples_split < 2 {
            return Err(RfError::InvalidMinSamplesSplit {
                min_samples_split: self.min_samples_split,
            });
        }

        if self.min_samples_leaf < 1 {
            return Err(RfError::InvalidMinSamplesLeaf {
                min_samples_leaf: self.min_samples_leaf,
            });
        }

        let max_features = self.max_features.unwrap_or(n_features);
        if max_features == 0 || max_features > n_features {
            return Err(RfError::InvalidMaxFeatures {
                max_features,
                n_features,
            });
        }

        // --- Derived values ---
        let n_classes = labels.iter().max().copied().unwrap_or(0) + 1;

        debug!(
            n_samples = n_samples,
            n_features = n_features,
            n_classes = n_classes,
            max_features = max_features,
            "fitting decision tree"
        );

        // Convert to column-major layout for find_best_split.
        let col_features: Vec<Vec<f64>> = (0..n_features)
            .map(|feat_idx| features.iter().map(|row| row[feat_idx]).collect())
            .collect();

        // Pre-compute histogram bins when using the Histogram split method.
        let bins: Option<FeatureBins> = match self.split_method {
            SplitMethod::Histogram { n_bins } => Some(FeatureBins::build(&col_features, n_bins)),
            _ => None,
        };

        let sample_indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut arena: Vec<Node> = Vec::new();

        let root = build_tree(
            &col_features,
            labels,
            &sample_indices,
            n_classes,
            self,
            0,
            &mut rng,
            &mut arena,
            max_features,
            bins.as_ref(),
        );

        debug!(
            root_index = root.index(),
            n_nodes = arena.len(),
            "decision tree built"
        );

        Ok(DecisionTree {
            nodes: arena,
            n_features,
            n_classes,
        })
    }
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively build the arena-based decision tree.
///
/// Returns the [`NodeIndex`] of the node just created in `arena`.
#[allow(clippy::too_many_arguments)]
fn build_tree(
    col_features: &[Vec<f64>],
    labels: &[usize],
    sample_indices: &[usize],
    n_classes: usize,
    config: &DecisionTreeConfig,
    depth: usize,
    rng: &mut ChaCha8Rng,
    arena: &mut Vec<Node>,
    max_features: usize,
    bins: Option<&FeatureBins>,
) -> NodeIndex {
    let n_samples = sample_indices.len();

    // Accumulate class counts.
    let mut class_counts = vec![0usize; n_classes];
    for &si in sample_indices {
        class_counts[labels[si]] += 1;
    }

    let impurity = config.criterion.impurity(&class_counts, n_samples);

    // Determine the majority-class prediction and normalized distribution.
    let make_leaf = |arena: &mut Vec<Node>| -> NodeIndex {
        let total = n_samples as f64;
        let distribution: Vec<f64> = class_counts.iter().map(|&c| c as f64 / total).collect();
        let prediction = class_counts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let idx = arena.len();
        arena.push(Node::Leaf {
            prediction,
            distribution,
            impurity,
            n_samples,
        });
        NodeIndex::new(idx)
    };

    // Stopping conditions → leaf.
    let depth_exceeded = config
        .max_depth
        .is_some_and(|max_d| depth >= max_d);
    let too_few = n_samples < config.min_samples_split;
    let pure = impurity.value() == 0.0;

    if too_few || pure || depth_exceeded {
        return make_leaf(arena);
    }

    // Try to find a split.
    let split_result = find_split_with_bins(
        col_features,
        labels,
        sample_indices,
        n_classes,
        &config.criterion,
        &config.split_method,
        bins,
        max_features,
        config.min_samples_leaf,
        rng,
    );

    let split = match split_result {
        Some(s) => s,
        None => return make_leaf(arena),
    };

    // Arena pattern: reserve index, recurse, then overwrite with the split.
    let node_idx = arena.len();
    // Push a temporary placeholder so children can reference valid indices.
    arena.push(Node::Leaf {
        prediction: 0,
        distribution: vec![0.0; n_classes],
        impurity,
        n_samples,
    });

    let left_idx = build_tree(
        col_features,
        labels,
        &split.left_indices,
        n_classes,
        config,
        depth + 1,
        rng,
        arena,
        max_features,
        bins,
    );

    let right_idx = build_tree(
        col_features,
        labels,
        &split.right_indices,
        n_classes,
        config,
        depth + 1,
        rng,
        arena,
        max_features,
        bins,
    );

    arena[node_idx] = Node::Split {
        feature: split.feature,
        threshold: split.threshold,
        left: left_idx,
        right: right_idx,
        impurity,
        n_samples,
        impurity_decrease: split.impurity_decrease,
    };

    NodeIndex::new(node_idx)
}

/// A fitted CART decision tree.
///
/// Stored as an arena-based `Vec<Node>` with index references for
/// cache-friendly traversal and trivial serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTree {
    pub(crate) nodes: Vec<Node>,
    pub(crate) n_features: usize,
    pub(crate) n_classes: usize,
}

impl DecisionTree {
    /// Predict the class label for a single sample.
    ///
    /// Traverses from the root (index 0): at each `Split`, goes left when
    /// `sample[feature] <= threshold`, right otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] when `sample.len() != n_features`.
    pub fn predict(&self, sample: &[f64]) -> Result<usize, RfError> {
        if sample.len() != self.n_features {
            return Err(RfError::PredictionFeatureMismatch {
                expected: self.n_features,
                got: sample.len(),
            });
        }
        let leaf = self.traverse(sample);
        match &self.nodes[leaf] {
            Node::Leaf { prediction, .. } => Ok(*prediction),
            Node::Split { .. } => unreachable!("traverse always ends at a leaf"),
        }
    }

    /// Return the class probability distribution for a single sample.
    ///
    /// The returned `Vec` has length `n_classes`, summing to 1.0.
    ///
    /// # Errors
    ///
    /// Returns [`RfError::PredictionFeatureMismatch`] when `sample.len() != n_features`.
    pub fn predict_proba(&self, sample: &[f64]) -> Result<Vec<f64>, RfError> {
        if sample.len() != self.n_features {
            return Err(RfError::PredictionFeatureMismatch {
                expected: self.n_features,
                got: sample.len(),
            });
        }
        let leaf = self.traverse(sample);
        match &self.nodes[leaf] {
            Node::Leaf { distribution, .. } => Ok(distribution.clone()),
            Node::Split { .. } => unreachable!("traverse always ends at a leaf"),
        }
    }

    /// Compute Mean Decrease in Impurity (MDI) feature importances.
    ///
    /// For each `Split` node, the `impurity_decrease` is accumulated by
    /// feature index, then the totals are normalized so they sum to 1.0.
    /// Returns a `Vec` of length `n_features`; all zeros when the tree is
    /// a single leaf.
    #[must_use]
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut totals = vec![0.0f64; self.n_features];
        for node in &self.nodes {
            if let Node::Split {
                feature,
                impurity_decrease,
                ..
            } = node
            {
                totals[feature.index()] += impurity_decrease;
            }
        }
        let sum: f64 = totals.iter().sum();
        if sum > 0.0 {
            totals.iter_mut().for_each(|v| *v /= sum);
        }
        totals
    }

    /// Return the total number of nodes in the tree (both splits and leaves).
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of leaf nodes.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Return the maximum depth of the tree.
    ///
    /// A single-node tree (just a root leaf) has depth 0.
    /// Uses an iterative BFS approach.
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        // BFS: (node_index, current_depth)
        let mut max_depth = 0usize;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((0usize, 0usize));

        while let Some((node_idx, d)) = queue.pop_front() {
            match &self.nodes[node_idx] {
                Node::Leaf { .. } => {
                    if d > max_depth {
                        max_depth = d;
                    }
                }
                Node::Split { left, right, .. } => {
                    queue.push_back((left.index(), d + 1));
                    queue.push_back((right.index(), d + 1));
                }
            }
        }

        max_depth
    }

    /// Traverse the tree from the root and return the arena index of the leaf.
    fn traverse(&self, sample: &[f64]) -> usize {
        let mut idx = 0usize;
        loop {
            match &self.nodes[idx] {
                Node::Leaf { .. } => return idx,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    if sample[feature.index()] <= *threshold {
                        idx = left.index();
                    } else {
                        idx = right.index();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dataset_error() {
        let features: Vec<Vec<f64>> = vec![];
        let labels: Vec<usize> = vec![];
        let err = DecisionTreeConfig::new().fit(&features, &labels).unwrap_err();
        assert!(matches!(err, RfError::EmptyDataset));
    }

    #[test]
    fn pure_dataset_single_leaf() {
        // All same label → single leaf node
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![0, 0, 0];
        let tree = DecisionTreeConfig::new().fit(&features, &labels).unwrap();
        assert_eq!(tree.n_nodes(), 1);
        assert_eq!(tree.n_leaves(), 1);
        assert_eq!(tree.predict(&[2.0, 3.0]).unwrap(), 0);
    }

    #[test]
    fn linearly_separable_correct_split() {
        // Feature 0: [1, 2, 3, 10, 11, 12], labels: [0, 0, 0, 1, 1, 1]
        let features = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
            vec![12.0, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTreeConfig::new()
            .with_seed(42)
            .fit(&features, &labels)
            .unwrap();
        assert_eq!(tree.predict(&[2.0, 0.0]).unwrap(), 0);
        assert_eq!(tree.predict(&[11.0, 0.0]).unwrap(), 1);
    }

    #[test]
    fn xor_needs_depth_at_least_2() {
        // XOR pattern requires at least 2 splits
        let features = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let labels = vec![0, 1, 1, 0];
        let tree = DecisionTreeConfig::new()
            .with_seed(42)
            .fit(&features, &labels)
            .unwrap();
        assert!(tree.depth() >= 2);
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTreeConfig::new().fit(&features, &labels).unwrap();
        let proba = tree.predict_proba(&[5.0]).unwrap();
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn feature_importances_sum_to_one() {
        let features = vec![
            vec![1.0, 100.0],
            vec![2.0, 200.0],
            vec![3.0, 300.0],
            vec![10.0, 100.0],
            vec![11.0, 200.0],
            vec![12.0, 300.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTreeConfig::new().fit(&features, &labels).unwrap();
        let importances = tree.feature_importances();
        let sum: f64 = importances.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
    }

    #[test]
    fn deterministic_with_same_seed() {
        let features = vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![3.0, 7.0],
            vec![10.0, 15.0],
            vec![11.0, 16.0],
            vec![12.0, 17.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree1 = DecisionTreeConfig::new()
            .with_seed(123)
            .fit(&features, &labels)
            .unwrap();
        let tree2 = DecisionTreeConfig::new()
            .with_seed(123)
            .fit(&features, &labels)
            .unwrap();
        // Same predictions on all training samples
        for sample in &features {
            assert_eq!(
                tree1.predict(sample).unwrap(),
                tree2.predict(sample).unwrap()
            );
        }
    }

    #[test]
    fn prediction_feature_mismatch() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![0, 1];
        let tree = DecisionTreeConfig::new().fit(&features, &labels).unwrap();
        let err = tree.predict(&[1.0]).unwrap_err();
        assert!(matches!(
            err,
            RfError::PredictionFeatureMismatch { expected: 2, got: 1 }
        ));
    }

    #[test]
    fn max_depth_limits_tree() {
        let features = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let labels = vec![0, 1, 1, 0];
        let tree = DecisionTreeConfig::new()
            .with_max_depth(Some(1))
            .with_seed(42)
            .fit(&features, &labels)
            .unwrap();
        assert!(tree.depth() <= 1);
    }

    #[test]
    fn feature_count_mismatch_error() {
        let features = vec![vec![1.0, 2.0], vec![3.0]]; // inconsistent
        let labels = vec![0, 1];
        let err = DecisionTreeConfig::new().fit(&features, &labels).unwrap_err();
        assert!(matches!(err, RfError::FeatureCountMismatch { .. }));
    }

    #[test]
    fn non_finite_value_error() {
        let features = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        let labels = vec![0, 1];
        let err = DecisionTreeConfig::new().fit(&features, &labels).unwrap_err();
        assert!(matches!(err, RfError::NonFiniteValue { .. }));
    }

    #[test]
    fn extra_trees_linearly_separable() {
        // Same dataset as linearly_separable_correct_split, but with ExtraTrees.
        let features = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
            vec![12.0, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTreeConfig::new()
            .with_split_method(SplitMethod::ExtraTrees)
            .with_seed(42)
            .fit(&features, &labels)
            .unwrap();
        assert_eq!(tree.predict(&[2.0, 0.0]).unwrap(), 0);
        assert_eq!(tree.predict(&[11.0, 0.0]).unwrap(), 1);
    }

    #[test]
    fn histogram_linearly_separable() {
        // Same dataset as linearly_separable_correct_split, but with Histogram splitting.
        let features = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 0.0],
            vec![11.0, 0.0],
            vec![12.0, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = DecisionTreeConfig::new()
            .with_split_method(SplitMethod::Histogram { n_bins: 8 })
            .with_seed(42)
            .fit(&features, &labels)
            .unwrap();
        assert_eq!(tree.predict(&[2.0, 0.0]).unwrap(), 0);
        assert_eq!(tree.predict(&[11.0, 0.0]).unwrap(), 1);
    }
}
