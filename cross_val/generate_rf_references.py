"""
Generate cross-validation reference data for narcissus-rf Random Forest.

Uses scikit-learn's RandomForestClassifier as ground truth.
Generates a synthetic dataset with make_classification and compares
predictions, feature importances, and accuracy.

Outputs a JSON file consumed by a Rust cross-validation example.
"""

import json

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ── Generate synthetic dataset ────────────────────────────────────────────
# Create a well-separated 3-class dataset with 5 informative features
# out of 10 total features

print("Generating synthetic dataset...")
X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=3.0,  # well-separated
    random_state=42,
    shuffle=True,
)

# Feature names
feature_names = [f"feature_{i}" for i in range(10)]

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Train sklearn RandomForestClassifier ──────────────────────────────────

print("Training scikit-learn RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=1,
)
clf.fit(X, y)

# Predictions on training data
predictions = clf.predict(X).tolist()
probas = clf.predict_proba(X).tolist()

# Feature importances
importances = clf.feature_importances_.tolist()

# Accuracy on training data
train_accuracy = float(clf.score(X, y))

# Feature importance ranking
importance_order = np.argsort(importances)[::-1].tolist()

print(f"  Training accuracy: {train_accuracy:.4f}")
print(f"  Top features: {[feature_names[i] for i in importance_order[:5]]}")
print(f"  Importances: {[round(importances[i], 4) for i in importance_order[:5]]}")

# ── Stratified 5-fold cross-validation ───────────────────────────────────

print("\nRunning 5-fold stratified cross-validation...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_fold_accuracies = []
cv_all_true = []
cv_all_pred = []
cv_fold_importances = []

for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    fold_clf = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )
    fold_clf.fit(X_train, y_train)

    fold_preds = fold_clf.predict(X_test)
    fold_acc = float(np.mean(fold_preds == y_test))
    cv_fold_accuracies.append(fold_acc)

    cv_all_true.extend(y_test.tolist())
    cv_all_pred.extend(fold_preds.tolist())

    cv_fold_importances.append(fold_clf.feature_importances_)
    print(f"  Fold {fold_idx}: accuracy={fold_acc:.4f}")

cv_mean_accuracy = float(np.mean(cv_fold_accuracies))
cv_std_accuracy = float(np.std(cv_fold_accuracies))  # ddof=0 (population std)

# Aggregated confusion matrix across all folds
n_classes = len(np.unique(y))
cv_cm = sklearn_confusion_matrix(cv_all_true, cv_all_pred, labels=list(range(n_classes)))
cv_overall_accuracy = float(np.trace(cv_cm) / np.sum(cv_cm))

# Per-class metrics from aggregated confusion matrix
cv_class_metrics = []
for c in range(n_classes):
    tp = int(cv_cm[c, c])
    fp = int(cv_cm[:, c].sum() - tp)
    fn_ = int(cv_cm[c, :].sum() - tp)
    support = tp + fn_
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / support if support > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    cv_class_metrics.append({
        "class": c,
        "precision": round(precision, 10),
        "recall": round(recall, 10),
        "f1": round(f1, 10),
        "support": support,
    })

# Average feature importances across folds, normalized
cv_avg_importances = np.mean(cv_fold_importances, axis=0)
cv_avg_importances = cv_avg_importances / cv_avg_importances.sum()
cv_importance_ranking = np.argsort(cv_avg_importances)[::-1].tolist()

print(f"  Mean accuracy: {cv_mean_accuracy:.4f} +/- {cv_std_accuracy:.4f}")
print(f"  Overall CM accuracy: {cv_overall_accuracy:.4f}")

# ── Assemble output ──────────────────────────────────────────────────────

output = {
    "description": "Random Forest cross-validation references (scikit-learn ground truth)",
    "python_versions": {"scikit-learn": "1.4"},
    "dataset": {
        "features": X.tolist(),
        "labels": y.tolist(),
        "feature_names": feature_names,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
    },
    "sklearn_rf": {
        "n_estimators": 100,
        "criterion": "gini",
        "max_features": "sqrt",
        "predictions": predictions,
        "probabilities": probas,
        "feature_importances": importances,
        "importance_ranking": importance_order,
        "train_accuracy": train_accuracy,
    },
    "sklearn_cv": {
        "n_folds": n_folds,
        "mean_accuracy": cv_mean_accuracy,
        "std_accuracy": cv_std_accuracy,
        "fold_accuracies": cv_fold_accuracies,
        "confusion_matrix": cv_cm.tolist(),
        "overall_accuracy": cv_overall_accuracy,
        "class_metrics": cv_class_metrics,
        "feature_importances": cv_avg_importances.tolist(),
        "importance_ranking": cv_importance_ranking,
    },
}

out_path = "/Users/nicolaslazaro/Desktop/work/narcissus/cross_val/rf_reference.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nReference data written to {out_path}")
