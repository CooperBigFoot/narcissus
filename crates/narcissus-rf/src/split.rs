use rand::Rng;

use crate::node::{FeatureIndex, Impurity};

/// Criterion for measuring the quality of a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SplitCriterion {
    /// Gini impurity: 1 - Σ(p_i²)
    Gini,
    /// Information entropy: -Σ(p_i · ln(p_i))
    Entropy,
}

impl SplitCriterion {
    /// Compute the impurity of a node from its class counts.
    ///
    /// Returns [`Impurity::new(0.0)`] when `n_samples` is zero (pure node).
    ///
    /// For `Gini`: `1 - Σ(p_i²)` where `p_i = count_i / n_samples`.
    /// For `Entropy`: `-Σ(p_i · ln(p_i))` summed only over classes where `p_i > 0`.
    #[must_use]
    pub fn impurity(&self, class_counts: &[usize], n_samples: usize) -> Impurity {
        if n_samples == 0 {
            return Impurity::new(0.0);
        }
        let n = n_samples as f64;
        let value = match self {
            SplitCriterion::Gini => {
                let sum_sq: f64 = class_counts
                    .iter()
                    .map(|&c| {
                        let p = c as f64 / n;
                        p * p
                    })
                    .sum();
                1.0 - sum_sq
            }
            SplitCriterion::Entropy => {
                -class_counts
                    .iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f64 / n;
                        p * p.ln()
                    })
                    .sum::<f64>()
            }
        };
        Impurity::new(value)
    }
}

/// Result of finding the best split for a node.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SplitResult {
    /// Feature used for the split.
    pub(crate) feature: FeatureIndex,
    /// Threshold value.
    pub(crate) threshold: f64,
    /// Weighted impurity decrease from this split (MDI formula).
    pub(crate) impurity_decrease: f64,
    /// Sample indices going to the left child.
    pub(crate) left_indices: Vec<usize>,
    /// Sample indices going to the right child.
    pub(crate) right_indices: Vec<usize>,
    /// Number of samples in left child.
    pub(crate) n_left: usize,
    /// Number of samples in right child.
    pub(crate) n_right: usize,
}

/// Find the best split among a random subset of features.
///
/// For each of `max_features` randomly chosen features, sorts the
/// `(value, label)` pairs, scans left-to-right with incremental
/// class count updates, and tracks the globally best split by
/// weighted impurity decrease.
///
/// Returns `None` when no valid split exists (all values identical,
/// or split would violate `min_samples_leaf`).
///
/// # Column-major layout
///
/// `features` is column-major: `features[feature_idx][sample_idx]`.
/// Each inner `Vec` contains all sample values for one feature column.
/// `sample_indices` are indices into these inner Vecs.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn find_best_split(
    features: &[Vec<f64>],
    labels: &[usize],
    sample_indices: &[usize],
    n_classes: usize,
    criterion: &SplitCriterion,
    max_features: usize,
    min_samples_leaf: usize,
    rng: &mut impl Rng,
) -> Option<SplitResult> {
    let n_features = features.len();
    let n_samples = sample_indices.len();

    if n_samples == 0 || n_features == 0 {
        return None;
    }

    // Build parent class counts.
    let mut parent_counts = vec![0usize; n_classes];
    for &si in sample_indices {
        parent_counts[labels[si]] += 1;
    }
    let parent_impurity = criterion.impurity(&parent_counts, n_samples);

    // Randomly shuffle feature indices and take up to max_features.
    let mut feature_order: Vec<usize> = (0..n_features).collect();
    // Partial Fisher-Yates: shuffle only the first `max_features` positions.
    let take = max_features.min(n_features);
    for i in 0..take {
        let j = rng.gen_range(i..n_features);
        feature_order.swap(i, j);
    }
    let selected_features = &feature_order[..take];

    let mut best_decrease = f64::NEG_INFINITY;
    let mut best: Option<(FeatureIndex, f64)> = None;

    for &feat_idx in selected_features {
        let feat_col = &features[feat_idx];

        // Collect (value, sample_index) pairs for this feature.
        let mut sorted: Vec<(f64, usize)> = sample_indices
            .iter()
            .map(|&si| (feat_col[si], si))
            .collect();
        sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        // Incremental scan: left grows from empty, right shrinks from full.
        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = parent_counts.clone();

        for i in 0..(n_samples - 1) {
            let (val_i, si) = sorted[i];
            let class_i = labels[si];

            // Move sample i from right to left.
            left_counts[class_i] += 1;
            right_counts[class_i] -= 1;

            let n_left = i + 1;
            let n_right = n_samples - n_left;

            // Skip if next value is identical (no valid boundary here).
            let val_next = sorted[i + 1].0;
            if val_i == val_next {
                continue;
            }

            // Enforce min_samples_leaf.
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }

            let left_impurity = criterion.impurity(&left_counts, n_left);
            let right_impurity = criterion.impurity(&right_counts, n_right);

            // MDI formula (matches scikit-learn).
            let decrease = (n_samples as f64) * parent_impurity.value()
                - (n_left as f64) * left_impurity.value()
                - (n_right as f64) * right_impurity.value();

            if decrease > best_decrease {
                best_decrease = decrease;
                let threshold = (val_i + val_next) / 2.0;
                best = Some((FeatureIndex::new(feat_idx), threshold));
            }
        }
    }

    let (best_feature, threshold) = best?;

    // Partition sample_indices into left/right.
    let feat_col = &features[best_feature.index()];
    let mut left_indices = Vec::with_capacity(n_samples / 2);
    let mut right_indices = Vec::with_capacity(n_samples / 2);
    for &si in sample_indices {
        if feat_col[si] <= threshold {
            left_indices.push(si);
        } else {
            right_indices.push(si);
        }
    }
    let n_left = left_indices.len();
    let n_right = right_indices.len();

    Some(SplitResult {
        feature: best_feature,
        threshold,
        impurity_decrease: best_decrease,
        left_indices,
        right_indices,
        n_left,
        n_right,
    })
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::{SplitCriterion, find_best_split};

    #[test]
    fn gini_pure() {
        let imp = SplitCriterion::Gini.impurity(&[10, 0, 0], 10);
        assert!((imp.value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gini_binary_balanced() {
        let imp = SplitCriterion::Gini.impurity(&[5, 5], 10);
        assert!((imp.value() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn gini_three_class_uniform() {
        let imp = SplitCriterion::Gini.impurity(&[100, 100, 100], 300);
        assert!((imp.value() - (1.0 - 3.0 * (1.0 / 3.0_f64).powi(2))).abs() < 1e-10);
    }

    #[test]
    fn entropy_pure() {
        let imp = SplitCriterion::Entropy.impurity(&[10, 0, 0], 10);
        assert!((imp.value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn entropy_binary_balanced() {
        let imp = SplitCriterion::Entropy.impurity(&[5, 5], 10);
        assert!((imp.value() - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn separable_data_finds_correct_split() {
        // Feature 0: [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        // Labels:    [0,   0,   0,    1,    1,    1  ]
        let features = vec![vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let sample_indices: Vec<usize> = (0..6).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = find_best_split(
            &features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            1,
            1,
            &mut rng,
        );

        let split = result.expect("should find a split");
        assert_eq!(split.feature.index(), 0);
        assert!(split.threshold > 3.0 && split.threshold < 10.0);
        assert_eq!(split.n_left, 3);
        assert_eq!(split.n_right, 3);
    }

    #[test]
    fn constant_feature_returns_none() {
        // All values are 5.0 — no valid split
        let features = vec![vec![5.0, 5.0, 5.0, 5.0]];
        let labels = vec![0, 0, 1, 1];
        let sample_indices: Vec<usize> = (0..4).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = find_best_split(
            &features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            1,
            1,
            &mut rng,
        );

        assert!(result.is_none());
    }

    #[test]
    fn min_samples_leaf_enforced() {
        // 2 samples, min_samples_leaf = 2 — can't split because each child
        // would have only 1 sample, violating the minimum of 2.
        let features = vec![vec![1.0, 10.0]];
        let labels = vec![0, 1];
        let sample_indices: Vec<usize> = (0..2).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = find_best_split(
            &features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            1,
            2,
            &mut rng,
        );

        assert!(result.is_none());
    }
}
