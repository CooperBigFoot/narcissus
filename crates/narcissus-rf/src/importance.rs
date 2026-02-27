//! Feature importance aggregation across trees.

/// A ranked feature with name, importance score, and rank.
#[derive(Debug, Clone)]
pub struct RankedFeature {
    /// Feature name.
    pub name: String,
    /// Normalized importance score (sums to 1.0 across all features).
    pub importance: f64,
    /// 1-based rank (1 = most important).
    pub rank: usize,
}

/// Aggregate per-tree feature importances into ranked features.
///
/// Sums importances across all trees, normalizes to sum to 1.0,
/// sorts descending by importance, and assigns 1-based ranks.
pub(crate) fn aggregate_importances(
    per_tree: &[Vec<f64>],
    names: &[String],
) -> Vec<RankedFeature> {
    if per_tree.is_empty() || names.is_empty() {
        return vec![];
    }

    let n_features = names.len();
    let mut totals = vec![0.0f64; n_features];

    for tree_imp in per_tree {
        for (i, &val) in tree_imp.iter().enumerate() {
            if i < n_features {
                totals[i] += val;
            }
        }
    }

    let sum: f64 = totals.iter().sum();
    if sum > 0.0 {
        totals.iter_mut().for_each(|v| *v /= sum);
    }

    let mut features: Vec<RankedFeature> = names
        .iter()
        .zip(totals.iter())
        .map(|(name, &importance)| RankedFeature {
            name: name.clone(),
            importance,
            rank: 0, // will be set after sorting
        })
        .collect();

    features.sort_by(|a, b| b.importance.total_cmp(&a.importance));

    for (i, feat) in features.iter_mut().enumerate() {
        feat.rank = i + 1;
    }

    features
}
