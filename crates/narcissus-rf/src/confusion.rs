//! Confusion matrix and per-class classification metrics.

use std::fmt;

use crate::error::RfError;

/// A confusion matrix for multi-class classification.
///
/// Entry `matrix[true_class][predicted_class]` counts how many samples
/// with true label `true_class` were predicted as `predicted_class`.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    matrix: Vec<Vec<usize>>,
    n_classes: usize,
}

/// Per-class precision, recall, and F1 score.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    /// The class index.
    pub class: usize,
    /// Precision: TP / (TP + FP). 0.0 if no predictions for this class.
    pub precision: f64,
    /// Recall: TP / (TP + FN). 0.0 if no true samples for this class.
    pub recall: f64,
    /// F1: 2 * precision * recall / (precision + recall). 0.0 if both are zero.
    pub f1: f64,
    /// Number of true samples in this class.
    pub support: usize,
}

impl ConfusionMatrix {
    /// Build a confusion matrix from true and predicted labels.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`RfError::EmptyDataset`] | Zero labels provided |
    pub fn from_labels(
        true_labels: &[usize],
        predicted: &[usize],
        n_classes: usize,
    ) -> Result<Self, RfError> {
        if true_labels.is_empty() {
            return Err(RfError::EmptyDataset);
        }
        let mut matrix = vec![vec![0usize; n_classes]; n_classes];
        for (&t, &p) in true_labels.iter().zip(predicted.iter()) {
            matrix[t][p] += 1;
        }
        Ok(Self { matrix, n_classes })
    }

    /// Overall accuracy: proportion of correct predictions.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.n_classes).map(|i| self.matrix[i][i]).sum();
        let total: usize = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }

    /// Per-class precision, recall, F1, and support.
    #[must_use]
    pub fn class_metrics(&self) -> Vec<ClassMetrics> {
        (0..self.n_classes)
            .map(|c| {
                let tp = self.matrix[c][c];
                let fp: usize = (0..self.n_classes)
                    .filter(|&i| i != c)
                    .map(|i| self.matrix[i][c])
                    .sum();
                let fn_: usize = (0..self.n_classes)
                    .filter(|&j| j != c)
                    .map(|j| self.matrix[c][j])
                    .sum();
                let support = tp + fn_;
                let precision = if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
                let recall = if support == 0 {
                    0.0
                } else {
                    tp as f64 / support as f64
                };
                let f1 = if precision + recall == 0.0 {
                    0.0
                } else {
                    2.0 * precision * recall / (precision + recall)
                };
                ClassMetrics {
                    class: c,
                    precision,
                    recall,
                    f1,
                    support,
                }
            })
            .collect()
    }

    /// Return the underlying matrix rows.
    #[must_use]
    pub fn as_rows(&self) -> &[Vec<usize>] {
        &self.matrix
    }

    /// Return the number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header row
        write!(f, "{:>8}", "")?;
        for j in 0..self.n_classes {
            write!(f, " pred_{j:>3}")?;
        }
        writeln!(f)?;

        // Data rows
        for (i, row) in self.matrix.iter().enumerate() {
            write!(f, "true_{i:>3}")?;
            for val in row {
                write!(f, " {val:>7}")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions() {
        let true_labels = vec![0, 0, 1, 1, 2, 2];
        let predicted = vec![0, 0, 1, 1, 2, 2];
        let cm = ConfusionMatrix::from_labels(&true_labels, &predicted, 3).unwrap();
        assert!((cm.accuracy() - 1.0).abs() < f64::EPSILON);

        for m in cm.class_metrics() {
            assert!((m.precision - 1.0).abs() < f64::EPSILON);
            assert!((m.recall - 1.0).abs() < f64::EPSILON);
            assert!((m.f1 - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn known_confusion_matrix() {
        // True: [0,0,0, 1,1,1, 2,2,2]
        // Pred: [0,0,1, 1,1,2, 2,2,0]
        let true_labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let predicted = vec![0, 0, 1, 1, 1, 2, 2, 2, 0];
        let cm = ConfusionMatrix::from_labels(&true_labels, &predicted, 3).unwrap();

        // Class 0: TP=2, FP=1(from class2), FN=1(to class1)
        // Class 1: TP=2, FP=1(from class0), FN=1(to class2)
        // Class 2: TP=2, FP=1(from class1), FN=1(to class0)
        let metrics = cm.class_metrics();

        // precision for class 0: 2/(2+1) = 0.667
        assert!((metrics[0].precision - 2.0 / 3.0).abs() < 1e-10);
        // recall for class 0: 2/(2+1) = 0.667
        assert!((metrics[0].recall - 2.0 / 3.0).abs() < 1e-10);
        // support for class 0: 3
        assert_eq!(metrics[0].support, 3);

        // Overall accuracy: 6/9 = 0.667
        assert!((cm.accuracy() - 6.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn empty_labels_error() {
        let err = ConfusionMatrix::from_labels(&[], &[], 3).unwrap_err();
        assert!(matches!(err, RfError::EmptyDataset));
    }

    #[test]
    fn display_formatting() {
        let cm = ConfusionMatrix::from_labels(&[0, 1], &[0, 1], 2).unwrap();
        let output = format!("{cm}");
        assert!(output.contains("pred_"));
        assert!(output.contains("true_"));
    }

    #[test]
    fn as_rows_returns_matrix() {
        let true_labels = vec![0, 0, 1, 1];
        let predicted = vec![0, 1, 0, 1];
        let cm = ConfusionMatrix::from_labels(&true_labels, &predicted, 2).unwrap();
        let rows = cm.as_rows();
        assert_eq!(rows[0], vec![1, 1]); // true 0: 1 correct, 1 misclassified as 1
        assert_eq!(rows[1], vec![1, 1]); // true 1: 1 misclassified as 0, 1 correct
    }

    #[test]
    fn zero_support_class_metrics() {
        // Class 2 has no true samples — precision/recall/f1 should be 0
        let true_labels = vec![0, 0, 1, 1];
        let predicted = vec![0, 0, 1, 1];
        let cm = ConfusionMatrix::from_labels(&true_labels, &predicted, 3).unwrap();
        let metrics = cm.class_metrics();
        assert_eq!(metrics[2].support, 0);
        assert!((metrics[2].recall - 0.0).abs() < f64::EPSILON);
    }
}
