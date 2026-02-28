//! Histogram-based splitting for decision trees.
//!
//! Quantile binning pre-computes per-feature bin edges once, then each split
//! candidate evaluation runs in O(n) for binning + O(B) for scanning bins,
//! avoiding the O(n log n) sort of exact CART.

use rand::Rng;

use crate::node::FeatureIndex;
use crate::split::{SplitCriterion, SplitResult};

/// Pre-computed quantile bin edges for all features.
#[derive(Debug, Clone)]
pub(crate) struct FeatureBins {
    /// Bin edges per feature. `edges[feat_idx]` has length `n_bins - 1`.
    /// Values <= edges[0] go into bin 0, values in (edges[i-1], edges[i]] go into bin i, etc.
    edges: Vec<Vec<f64>>,
    /// Number of bins requested at build time (same for all features).
    #[allow(dead_code)]
    n_bins: usize,
}

impl FeatureBins {
    /// Build quantile-based bin edges from column-major features.
    ///
    /// `col_features[feat_idx][sample_idx]` — column-major layout.
    /// For each feature, sorts values, computes quantile edges at
    /// `1/n_bins, 2/n_bins, ...` positions, deduplicates adjacent equal edges.
    /// Features with all identical values get zero edges (treated as constant).
    pub(crate) fn build(col_features: &[Vec<f64>], n_bins: usize) -> Self {
        let edges = col_features
            .iter()
            .map(|col| {
                if col.is_empty() {
                    return Vec::new();
                }

                let mut sorted = col.clone();
                sorted.sort_unstable_by(|a, b| a.total_cmp(b));

                let n = sorted.len();

                // Constant feature — no edges.
                if sorted[0] == sorted[n - 1] {
                    return Vec::new();
                }

                // Compute quantile edge positions at 1/n_bins, 2/n_bins, ..., (n_bins-1)/n_bins.
                let mut raw_edges: Vec<f64> = (1..n_bins)
                    .map(|k| {
                        let pos = (k as f64 / n_bins as f64) * (n - 1) as f64;
                        let lo = pos.floor() as usize;
                        let hi = (lo + 1).min(n - 1);
                        let frac = pos - lo as f64;
                        sorted[lo] + frac * (sorted[hi] - sorted[lo])
                    })
                    .collect();

                // Deduplicate adjacent equal edges.
                raw_edges.dedup_by(|a, b| *a == *b);

                // Clamp edges to be strictly within (min, max) — drop any that
                // equal the minimum (they'd put all samples in right) or equal
                // the maximum (they'd put all samples in left).
                raw_edges.retain(|&e| e > sorted[0] && e < sorted[n - 1]);

                raw_edges
            })
            .collect();

        Self { edges, n_bins }
    }

    /// Return the bin index for a given value in a given feature.
    ///
    /// Uses binary search on the edges. Returns a value in `[0, actual_bins)`.
    pub(crate) fn bin_index(&self, feat_idx: usize, value: f64) -> usize {
        let edges = &self.edges[feat_idx];
        // partition_point returns the number of edges strictly less than `value`
        // (i.e., the first index where edges[i] >= value).
        // We want the bin index such that values <= edge[bin_idx] go left.
        // Binary search: find first edge > value → that is the bin index.
        edges.partition_point(|&e| e < value)
    }

    /// Return the number of actual bins for a feature (`edges.len() + 1`,
    /// or 0 if the feature is constant).
    pub(crate) fn n_bins_for_feature(&self, feat_idx: usize) -> usize {
        let n_edges = self.edges[feat_idx].len();
        if n_edges == 0 {
            0
        } else {
            n_edges + 1
        }
    }

    /// Return the threshold value for a given bin boundary.
    ///
    /// The threshold is the edge value at `bin_idx` (splitting at bin_idx means
    /// values with bin <= bin_idx go left, others go right).
    pub(crate) fn threshold(&self, feat_idx: usize, bin_idx: usize) -> f64 {
        self.edges[feat_idx][bin_idx]
    }
}

/// Find the best split using histogram-based evaluation.
///
/// For each of `max_features` randomly chosen features:
/// 1. Bin each sample's feature value into its pre-computed bin (O(n))
/// 2. Accumulate per-bin class counts (O(n))
/// 3. Scan bins left-to-right to find the best split point (O(B))
///
/// The threshold is the actual feature value at the bin boundary (not the bin index).
///
/// Returns `None` when no valid split exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_histogram_split(
    col_features: &[Vec<f64>],
    labels: &[usize],
    sample_indices: &[usize],
    n_classes: usize,
    criterion: &SplitCriterion,
    bins: &FeatureBins,
    max_features: usize,
    min_samples_leaf: usize,
    rng: &mut impl Rng,
) -> Option<SplitResult> {
    let n_features = col_features.len();
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
    let take = max_features.min(n_features);
    for i in 0..take {
        let j = rng.gen_range(i..n_features);
        feature_order.swap(i, j);
    }
    let selected_features = &feature_order[..take];

    let mut best_decrease = f64::NEG_INFINITY;
    let mut best: Option<(FeatureIndex, f64)> = None;

    for &feat_idx in selected_features {
        let actual_bins = bins.n_bins_for_feature(feat_idx);

        // Constant feature — skip.
        if actual_bins == 0 {
            continue;
        }

        // Accumulate per-bin class counts.
        // bin_counts[bin_idx][class_idx]
        let mut bin_counts: Vec<Vec<usize>> = vec![vec![0usize; n_classes]; actual_bins];
        let feat_col = &col_features[feat_idx];
        for &si in sample_indices {
            let bin = bins.bin_index(feat_idx, feat_col[si]);
            // bin_index returns index in [0, edges.len()] = [0, actual_bins - 1]
            // so clamping is safe in practice, but cap it to be defensive.
            let bin = bin.min(actual_bins - 1);
            bin_counts[bin][labels[si]] += 1;
        }

        // Scan bins left-to-right: maintain cumulative left/right class counts.
        // A split "at bin boundary b" means bins [0..=b] go left, bins [b+1..] go right.
        // There are (actual_bins - 1) = edges.len() split points.
        let n_edges = bins.edges[feat_idx].len();

        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = parent_counts.clone();

        let mut n_left = 0usize;
        let mut n_right = n_samples;

        for (split_bin, bin_cls_counts) in bin_counts.iter().enumerate().take(n_edges) {
            // Move bin `split_bin` from right to left.
            let bin_total: usize = bin_cls_counts.iter().sum();
            n_left += bin_total;
            n_right -= bin_total;

            for (cls, &cnt) in bin_cls_counts.iter().enumerate() {
                left_counts[cls] += cnt;
                right_counts[cls] -= cnt;
            }

            // Skip if either side is empty (degenerate split).
            if n_left == 0 || n_right == 0 {
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
                let threshold = bins.threshold(feat_idx, split_bin);
                best = Some((FeatureIndex::new(feat_idx), threshold));
            }
        }
    }

    let (best_feature, threshold) = best?;

    // Partition sample_indices into left/right using the selected threshold.
    let feat_col = &col_features[best_feature.index()];
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

    use crate::split::{SplitCriterion, find_best_split};

    use super::{FeatureBins, find_histogram_split};

    /// Six-sample perfectly-separable dataset used across several tests.
    fn separable_features() -> Vec<Vec<f64>> {
        // Column-major: one feature with 6 samples.
        vec![vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]]
    }

    fn separable_labels() -> Vec<usize> {
        vec![0, 0, 0, 1, 1, 1]
    }

    #[test]
    fn histogram_separable_data() {
        let col_features = separable_features();
        let labels = separable_labels();
        let sample_indices: Vec<usize> = (0..6).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let bins = FeatureBins::build(&col_features, 8);
        let result = find_histogram_split(
            &col_features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            &bins,
            1,
            1,
            &mut rng,
        );

        let split = result.expect("should find a split on separable data");
        assert_eq!(split.feature.index(), 0);
        // Threshold must lie in the gap (3.0, 10.0].
        assert!(
            split.threshold > 3.0 && split.threshold <= 10.0,
            "threshold = {}",
            split.threshold
        );
        assert_eq!(split.n_left, 3);
        assert_eq!(split.n_right, 3);
    }

    #[test]
    fn histogram_constant_feature_skip() {
        // All values identical → no valid split.
        let col_features = vec![vec![5.0, 5.0, 5.0, 5.0]];
        let labels = vec![0, 0, 1, 1];
        let sample_indices: Vec<usize> = (0..4).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let bins = FeatureBins::build(&col_features, 8);
        let result = find_histogram_split(
            &col_features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            &bins,
            1,
            1,
            &mut rng,
        );

        assert!(result.is_none(), "constant feature should produce no split");
    }

    #[test]
    fn histogram_accuracy_within_tolerance_of_exact() {
        // Compare impurity_decrease of histogram vs exact split on the same separable data.
        // Histogram may not be optimal due to quantization, but should be within reason.
        let col_features = separable_features();
        let labels = separable_labels();
        let sample_indices: Vec<usize> = (0..6).collect();

        let bins = FeatureBins::build(&col_features, 16);

        let mut rng_hist = ChaCha8Rng::seed_from_u64(42);
        let hist_split = find_histogram_split(
            &col_features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            &bins,
            1,
            1,
            &mut rng_hist,
        )
        .expect("histogram should find a split");

        let mut rng_exact = ChaCha8Rng::seed_from_u64(42);
        let exact_split = find_best_split(
            &col_features,
            &labels,
            &sample_indices,
            2,
            &SplitCriterion::Gini,
            1,
            1,
            &mut rng_exact,
        )
        .expect("exact should find a split");

        // Both should achieve perfect separation (impurity_decrease close to exact).
        let diff = (hist_split.impurity_decrease - exact_split.impurity_decrease).abs();
        assert!(
            diff < 0.5,
            "impurity_decrease too far: hist={}, exact={}",
            hist_split.impurity_decrease,
            exact_split.impurity_decrease
        );
    }

    #[test]
    fn bins_build_correct_edges() {
        // Uniform values 1..=10 with 5 bins → 4 edges at quantile positions.
        let col_features = vec![(1..=10).map(|x| x as f64).collect::<Vec<_>>()];
        let bins = FeatureBins::build(&col_features, 5);

        // Should produce 4 edges (n_bins - 1).
        let n_edges = bins.edges[0].len();
        assert!(
            n_edges > 0 && n_edges <= 4,
            "expected 1-4 edges, got {n_edges}"
        );

        // Edges must be strictly increasing.
        let edges = &bins.edges[0];
        for w in edges.windows(2) {
            assert!(
                w[0] < w[1],
                "edges not strictly increasing: {:?}",
                edges
            );
        }

        // All edges must lie strictly within (min, max) = (1.0, 10.0).
        for &e in edges {
            assert!(e > 1.0 && e < 10.0, "edge {e} out of (1.0, 10.0)");
        }
    }

    #[test]
    fn bin_index_correctness() {
        // Feature values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        // 4 bins → edges roughly at 3.25, 5.5, 7.75 (quantile interpolation).
        let col_features = vec![(1..=10).map(|x| x as f64).collect::<Vec<_>>()];
        let bins = FeatureBins::build(&col_features, 4);

        // Minimum value (1.0) must map to bin 0.
        let bin_min = bins.bin_index(0, 1.0);
        assert_eq!(bin_min, 0, "min value should be in bin 0");

        // Maximum value (10.0) must map to the last bin.
        let n_actual = bins.n_bins_for_feature(0);
        let bin_max = bins.bin_index(0, 10.0);
        assert_eq!(
            bin_max,
            n_actual - 1,
            "max value should be in last bin ({}), got {}",
            n_actual - 1,
            bin_max
        );

        // A value in the middle must map to a valid bin index.
        let bin_mid = bins.bin_index(0, 5.5);
        assert!(bin_mid < n_actual, "mid bin index {bin_mid} out of range");
    }
}
