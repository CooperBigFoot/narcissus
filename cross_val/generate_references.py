"""
Generate cross-validation reference data for narcissus-cluster K-means.

Uses tslearn's TimeSeriesKMeans (DTW + DBA) as ground truth.
DTW distance correctness is already validated separately — this script
focuses on clustering behavior: assignments, inertia, centroids, and
elbow curve properties.

Outputs a JSON file consumed by a Rust cross-validation example.
"""

import json

import numpy as np
from dtaidistance import dtw as dtai_dtw
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

# ── Test data: 9 series in 3 tight clusters ─────────────────────────────────

SERIES = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.0, 0.0],
    [5.0, 5.0, 5.0, 5.0],
    [5.1, 5.0, 5.0, 5.0],
    [5.0, 5.1, 5.0, 5.0],
    [10.0, 10.0, 10.0, 10.0],
    [10.1, 10.0, 10.0, 10.0],
    [10.0, 10.1, 10.0, 10.0],
]

dataset = to_time_series_dataset(SERIES)

# ── 1. DBA centroids per known group ────────────────────────────────────────
# Given perfect assignments, what centroids does DBA produce?

print("Computing DBA centroids per group...")
groups = {
    "group_0": SERIES[0:3],
    "group_1": SERIES[3:6],
    "group_2": SERIES[6:9],
}

dba_centroids = {}
for name, members in groups.items():
    ds = to_time_series_dataset(members)
    centroid = dtw_barycenter_averaging(ds, max_iter=50, tol=1e-8)
    dba_centroids[name] = centroid.flatten().tolist()

# ── 2. K-means k=3 ──────────────────────────────────────────────────────────
# The clusters are so well-separated that any correct implementation must
# produce the same 3-way partition regardless of initialization.

print("Running tslearn TimeSeriesKMeans (k=3, 10 inits)...")
km3 = TimeSeriesKMeans(
    n_clusters=3,
    metric="dtw",
    n_init=10,
    max_iter=75,
    tol=1e-4,
    random_state=42,
    verbose=0,
)
labels3 = km3.fit_predict(dataset)

# Build partition: group indices by cluster label
partition3 = {}
for i, c in enumerate(labels3):
    c = int(c)
    partition3.setdefault(c, []).append(i)

# Manually recompute inertia as sum of squared DTW distances to centroid
inertia3 = 0.0
for i, s in enumerate(SERIES):
    c = int(labels3[i])
    centroid = km3.cluster_centers_[c].flatten()
    d = dtai_dtw.distance(np.array(s), np.array(centroid))
    inertia3 += d**2

# ── 3. K-means k=1 (trivial) ────────────────────────────────────────────────

print("Running k=1 baseline...")
km1 = TimeSeriesKMeans(
    n_clusters=1,
    metric="dtw",
    n_init=1,
    max_iter=75,
    tol=1e-4,
    random_state=42,
    verbose=0,
)
km1.fit(dataset)
k1_centroid = km1.cluster_centers_[0].flatten().tolist()

k1_inertia = 0.0
for s in SERIES:
    d = dtai_dtw.distance(np.array(s), np.array(k1_centroid))
    k1_inertia += d**2

# ── 4. K-means k=9 (k=n, should give ~0 inertia) ───────────────────────────

print("Running k=9 (k=n)...")
km9 = TimeSeriesKMeans(
    n_clusters=9,
    metric="dtw",
    n_init=1,
    max_iter=75,
    tol=1e-4,
    random_state=42,
    verbose=0,
)
km9.fit(dataset)
k9_inertia = float(km9.inertia_)

# ── 5. Elbow sweep k=1..9 ───────────────────────────────────────────────────

print("Running elbow sweep k=1..9...")
elbow_data = []
for k in range(1, 10):
    km_k = TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        n_init=5,
        max_iter=75,
        tol=1e-4,
        random_state=42,
        verbose=0,
    )
    km_k.fit(dataset)
    elbow_data.append({"k": k, "inertia": float(km_k.inertia_)})

# ── Assemble output ─────────────────────────────────────────────────────────

output = {
    "description": "K-means clustering cross-validation references (tslearn ground truth)",
    "python_versions": {"tslearn": "0.8.0", "dtaidistance": "2.4.0"},
    "series": SERIES,
    "dba_centroids": dba_centroids,
    "kmeans_k3": {
        "labels": [int(x) for x in labels3],
        "inertia": inertia3,
        "tslearn_inertia": float(km3.inertia_),
        "centroids": [
            km3.cluster_centers_[c].flatten().tolist() for c in range(3)
        ],
        "partition": {str(k): v for k, v in partition3.items()},
    },
    "kmeans_k1": {
        "inertia": k1_inertia,
        "centroid": k1_centroid,
    },
    "kmeans_k9": {
        "inertia": k9_inertia,
    },
    "elbow": elbow_data,
}

out_path = "/Users/nicolaslazaro/Desktop/work/narcissus/cross_val/reference.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nReference data written to {out_path}")
print(f"  K=3 partition: {partition3}")
print(f"  K=3 inertia (manual): {inertia3:.6f}")
print(f"  K=3 inertia (tslearn): {km3.inertia_:.6f}")
print(f"  K=1 inertia: {k1_inertia:.6f}")
print(f"  K=9 inertia: {k9_inertia:.6f}")
print(f"  Elbow: {[(e['k'], round(e['inertia'], 4)) for e in elbow_data]}")
