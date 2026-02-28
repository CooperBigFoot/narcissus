//! Permutation-based feature importance.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::forest::RandomForest;
use crate::tree::DecisionTree;

/// Permutation importance result for a single feature.
#[derive(Debug, Clone)]
pub struct PermutationImportance {
    /// Feature name.
    pub name: String,
    /// Mean accuracy drop when this feature is permuted.
    pub importance: f64,
    /// Standard deviation of the accuracy drop across trees.
    pub std: f64,
    /// Rank (1 = most important).
    pub rank: usize,
}

/// Compute baseline accuracy of a single tree on a set of OOB samples.
fn tree_oob_accuracy(
    tree: &DecisionTree,
    features: &[Vec<f64>],
    labels: &[usize],
    oob_indices: &[usize],
) -> f64 {
    if oob_indices.is_empty() {
        return 0.0;
    }
    let correct = oob_indices
        .iter()
        .filter(|&&idx| {
            tree.predict(&features[idx])
                .map(|pred| pred == labels[idx])
                .unwrap_or(false)
        })
        .count();
    correct as f64 / oob_indices.len() as f64
}

/// Compute accuracy of a single tree on OOB samples with one feature permuted.
fn tree_permuted_accuracy(
    tree: &DecisionTree,
    features: &[Vec<f64>],
    labels: &[usize],
    oob_indices: &[usize],
    feature_idx: usize,
    rng: &mut ChaCha8Rng,
) -> f64 {
    if oob_indices.is_empty() {
        return 0.0;
    }

    // Collect the feature values for the OOB samples and shuffle them.
    let mut permuted_values: Vec<f64> = oob_indices
        .iter()
        .map(|&idx| features[idx][feature_idx])
        .collect();
    permuted_values.shuffle(rng);

    // Predict with permuted feature column.
    let correct = oob_indices
        .iter()
        .zip(permuted_values.iter())
        .filter(|pair| {
            let idx = *pair.0;
            let permuted_val = *pair.1;
            let mut sample = features[idx].clone();
            sample[feature_idx] = permuted_val;
            tree.predict(&sample)
                .map(|pred| pred == labels[idx])
                .unwrap_or(false)
        })
        .count();
    correct as f64 / oob_indices.len() as f64
}

/// Compute permutation feature importance using per-tree OOB samples.
///
/// For each tree and each feature:
/// 1. Compute baseline OOB accuracy for this tree (predict on OOB samples)
/// 2. Permute the feature column among OOB samples
/// 3. Compute permuted OOB accuracy
/// 4. Importance = baseline_accuracy - permuted_accuracy
///
/// The final importance per feature is the mean across all trees.
/// Standard deviation is computed across trees (population std, ddof=0).
pub(crate) fn compute_permutation_importance(
    forest: &RandomForest,
    features: &[Vec<f64>],
    labels: &[usize],
    oob_indices_per_tree: &[Vec<usize>],
    feature_names: &[String],
    seed: u64,
) -> Vec<PermutationImportance> {
    let n_features = feature_names.len();

    // Collect per-tree, per-feature accuracy drops.
    // Only include trees with non-empty OOB sets.
    let mut drops: Vec<Vec<f64>> = Vec::new();

    for (tree_idx, (tree, oob_indices)) in forest
        .trees
        .iter()
        .zip(oob_indices_per_tree.iter())
        .enumerate()
    {
        if oob_indices.is_empty() {
            continue;
        }

        let baseline_acc = tree_oob_accuracy(tree, features, labels, oob_indices);

        let mut tree_drops = Vec::with_capacity(n_features);
        for feat_idx in 0..n_features {
            let rng_seed = seed
                .wrapping_add((tree_idx as u64).wrapping_mul(n_features as u64))
                .wrapping_add(feat_idx as u64);
            let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);

            let permuted_acc =
                tree_permuted_accuracy(tree, features, labels, oob_indices, feat_idx, &mut rng);
            tree_drops.push(baseline_acc - permuted_acc);
        }
        drops.push(tree_drops);
    }

    // If no trees had OOB samples, return all-zero importances.
    if drops.is_empty() {
        return feature_names
            .iter()
            .enumerate()
            .map(|(i, name)| PermutationImportance {
                name: name.clone(),
                importance: 0.0,
                std: 0.0,
                rank: i + 1,
            })
            .collect();
    }

    let n_valid_trees = drops.len() as f64;

    // Compute mean and std per feature.
    let mut results: Vec<PermutationImportance> = (0..n_features)
        .map(|feat_idx| {
            let values: Vec<f64> = drops.iter().map(|tree_drops| tree_drops[feat_idx]).collect();

            let mean = values.iter().sum::<f64>() / n_valid_trees;
            let variance =
                values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n_valid_trees;
            let std = variance.sqrt();

            PermutationImportance {
                name: feature_names[feat_idx].clone(),
                importance: mean,
                std,
                rank: 0, // will be set after sorting
            }
        })
        .collect();

    // Sort by importance descending and assign ranks.
    results.sort_by(|a, b| b.importance.total_cmp(&a.importance));
    for (i, result) in results.iter_mut().enumerate() {
        result.rank = i + 1;
    }

    results
}

#[cfg(test)]
mod tests {
    use crate::config::{OobMode, RandomForestConfig};

    /// Generate separable data where feature 0 is informative and feature 1 is noise.
    fn make_data() -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for i in 0..30 {
            features.push(vec![i as f64 * 0.1, 0.5]);
            labels.push(0);
        }
        for i in 0..30 {
            features.push(vec![10.0 + i as f64 * 0.1, 0.5]);
            labels.push(1);
        }
        for i in 0..30 {
            features.push(vec![20.0 + i as f64 * 0.1, 0.5]);
            labels.push(2);
        }
        let names = vec!["informative".to_string(), "noise".to_string()];
        (features, labels, names)
    }

    #[test]
    fn informative_feature_high_importance() {
        let (features, labels, names) = make_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_oob_mode(OobMode::Enabled)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();
        let perm_imp = result.permutation_importances(&features, &labels, 42);

        assert_eq!(perm_imp.len(), 2);
        // Feature 0 (informative) should have much higher importance than feature 1 (noise)
        let feat0 = perm_imp.iter().find(|p| p.name == "informative").unwrap();
        let feat1 = perm_imp.iter().find(|p| p.name == "noise").unwrap();
        assert!(
            feat0.importance > feat1.importance,
            "informative ({}) should have higher importance than noise ({})",
            feat0.importance,
            feat1.importance
        );
        assert!(
            feat0.importance > 0.1,
            "informative feature importance should be substantial: {}",
            feat0.importance
        );
    }

    #[test]
    fn constant_feature_near_zero() {
        let (features, labels, names) = make_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_oob_mode(OobMode::Enabled)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();
        let perm_imp = result.permutation_importances(&features, &labels, 42);

        let noise = perm_imp.iter().find(|p| p.name == "noise").unwrap();
        assert!(
            noise.importance.abs() < 0.3,
            "noise feature importance should be near zero: {}",
            noise.importance
        );
    }

    #[test]
    fn rank_order_consistent() {
        let (features, labels, names) = make_data();
        let config = RandomForestConfig::new(50)
            .unwrap()
            .with_oob_mode(OobMode::Enabled)
            .with_seed(42);
        let result = config.fit(&features, &labels, &names).unwrap();
        let perm_imp = result.permutation_importances(&features, &labels, 42);

        // Ranks should be 1 and 2
        let ranks: Vec<usize> = perm_imp.iter().map(|p| p.rank).collect();
        assert!(ranks.contains(&1));
        assert!(ranks.contains(&2));

        // Rank 1 should have the highest importance
        let rank1 = perm_imp.iter().find(|p| p.rank == 1).unwrap();
        let rank2 = perm_imp.iter().find(|p| p.rank == 2).unwrap();
        assert!(rank1.importance >= rank2.importance);
    }
}
