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
}

out_path = "/Users/nicolaslazaro/Desktop/work/narcissus/cross_val/rf_reference.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nReference data written to {out_path}")
