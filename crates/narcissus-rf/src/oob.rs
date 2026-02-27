//! Out-of-bag (OOB) evaluation for Random Forest.

use crate::error::RfError;
use crate::tree::DecisionTree;

/// Out-of-bag evaluation result.
#[derive(Debug, Clone)]
pub struct OobScore {
    /// OOB accuracy (fraction of correctly predicted OOB samples).
    pub accuracy: f64,
    /// OOB confusion matrix: `confusion_matrix[true][predicted]`.
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Number of samples that had at least one OOB tree.
    pub n_oob_samples: usize,
}

/// Compute out-of-bag predictions and accuracy.
///
/// For each sample, only trees where the sample was NOT in the bootstrap
/// are used for prediction (majority vote). Samples with no OOB trees
/// are skipped.
pub(crate) fn compute_oob(
    trees: &[DecisionTree],
    features: &[Vec<f64>],
    labels: &[usize],
    _n_features: usize,
    n_classes: usize,
    oob_indices_per_tree: &[Vec<usize>],
) -> Result<OobScore, RfError> {
    let n_samples = features.len();

    // For each sample, accumulate class votes from OOB trees.
    let mut oob_votes: Vec<Vec<usize>> = vec![vec![0; n_classes]; n_samples];
    let mut has_oob = vec![false; n_samples];

    for (tree_idx, oob_indices) in oob_indices_per_tree.iter().enumerate() {
        for &sample_idx in oob_indices {
            let pred = trees[tree_idx].predict(&features[sample_idx])?;
            oob_votes[sample_idx][pred] += 1;
            has_oob[sample_idx] = true;
        }
    }

    // Count OOB-evaluated samples.
    let n_oob_samples = has_oob.iter().filter(|&&h| h).count();
    if n_oob_samples == 0 {
        return Err(RfError::OobEvaluationFailed {
            reason: "no sample has any OOB tree".to_string(),
        });
    }

    // Build confusion matrix from OOB predictions.
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0usize;

    for (i, votes) in oob_votes.iter().enumerate() {
        if !has_oob[i] {
            continue;
        }
        let predicted = votes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        confusion[labels[i]][predicted] += 1;
        if predicted == labels[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / n_oob_samples as f64;

    Ok(OobScore {
        accuracy,
        confusion_matrix: confusion,
        n_oob_samples,
    })
}
