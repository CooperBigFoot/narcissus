# Narcissus

A high-performance Rust CLI for shape-based hydrological basin clustering and classification. Ported from [find-similar-basins](https://github.com/nicolaslazaro/find-similar-basins) (Python) with hand-rolled algorithms for maximum throughput.

Designed to be driven by an LLM agent — structured JSON output, deterministic execution, composable subcommands.

> **Based on:** Yang & Olivera (2023) — *"Classification of watersheds in the conterminous United States using shape-based time-series clustering and Random Forests."*

---

## Component Tracker

High-level capabilities that Narcissus must deliver. Status: `pending` | `in-progress` | `done` | `deferred`.

| # | Component | Description | Status | Notes |
|---|---|---|---|---|
| 1 | **DTW distance** | Dynamic Time Warping with configurable warping window (Sakoe-Chiba). The core distance metric. | `done` | Hand-rolled. Sakoe-Chiba band constraint (radius=2 weeks for 52-week series). Rolling-buffer fast path (112 bytes for n=52, r=2). Rayon-parallelized pairwise computation. Cross-validated against `dtaidistance` (Python). |
| 2 | **Barycenter averaging** | Compute a representative centroid for a group of time series under DTW alignment (DBA algorithm). | `done` | Hand-rolled. DBA algorithm (Petitjean et al. 2011). Iterative warping-path alignment to update centroid. Configurable convergence tolerance and max iterations. |
| 3 | **K-means clustering** | Cluster time series into k groups using DTW as the distance metric. Includes smart initialization (k-means++) and multi-restart. | `pending` | Hand-rolled. Depends on #1 and #2. |
| 4 | **Cluster count selection** | Run clustering across a range of k values and report fit quality (inertia) so the user or LLM can pick the best k (elbow method). | `pending` | Thin wrapper over #3. |
| 5 | **Random Forest classification** | Train a classifier that maps static basin attributes → cluster labels. Must support probability output (top-k predictions). | `pending` | Evaluate existing Rust crates (`smartcore`, `linfa-ensemble`) vs. hand-rolling. Decision deferred until clustering is done. |
| 6 | **Model evaluation** | Cross-validated accuracy, confusion matrix, feature importance ranking. | `pending` | Depends on #5. Stratified k-fold CV needed. |
| 7 | **Prediction** | Load a trained model, predict cluster membership with probabilities for new basins. | `pending` | Depends on #5. Top-k ranked output. |
| 8 | **Data ingestion** | Read time series and tabular attribute data from disk. Format TBD (CSV, NPY, Parquet, or custom). | `pending` | Keep format-agnostic internally — concrete readers are a separate concern. |
| 9 | **Validation** | Validate inputs at the boundary: shape, missing values, ID consistency, feature schema matching. | `pending` | Parse-don't-validate philosophy per CLAUDE.md. |
| 10 | **Result persistence** | Write cluster assignments, centroids, model artifacts, and evaluation metrics to disk. | `pending` | JSON for structured output. Tabular formats for large results. |
| 11 | **CLI** | Subcommand-based interface (`optimize`, `cluster`, `evaluate`, `predict`). JSON to stdout, diagnostics to stderr. | `pending` | `clap` derive API. |
| 12 | **Model serialization** | Save/load trained RF models in a Rust-native format. | `pending` | Depends on #5 — format determined by RF implementation choice. |

### Open Design Questions

| Question | Options | Status |
|---|---|---|
| RF implementation | Hand-roll / `smartcore` / `linfa-ensemble` / hybrid | **Deferred** — evaluate after clustering works |
| Input time series format | CSV / NPY / Parquet / custom binary | **Open** |
| Input attributes format | CSV / Parquet | **Open** |
| Centroid output format | Same as input / JSON / CSV | **Open** |
| Model binary format | Custom / bincode / messagepack | **Open** — depends on RF choice |

---

## Architecture

```
narcissus (workspace root)
├── Cargo.toml              # workspace manifest
├── crates/
│   ├── narcissus-dtw/      # DTW distance + DBA barycenter averaging
│   ├── narcissus-cluster/  # K-means loop, initialization, elbow optimization
│   ├── narcissus-rf/       # Random Forest: train, evaluate, predict
│   └── narcissus-io/       # File I/O, validation, serialization
└── src/
    └── main.rs             # CLI (clap subcommands)
```

### Crate Dependency Graph

```
narcissus (bin)
├── narcissus-cluster
│   └── narcissus-dtw
├── narcissus-rf
└── narcissus-io
```

- **narcissus-dtw** — Pure math, zero I/O. DTW distance computation and DBA.
- **narcissus-cluster** — Orchestrates clustering using DTW primitives.
- **narcissus-rf** — Independent of DTW. Operates on tabular (attribute → label) data.
- **narcissus-io** — All file formats, parsing, validation, serialization.
- **narcissus** (bin) — Thin CLI shell. Wires crates together behind `clap`.

---

## CLI Design

All commands emit structured JSON to **stdout**. Progress and diagnostics go to **stderr** via `tracing`. This makes every command pipeable and parseable by an LLM.

### `narcissus optimize`

Run clustering for a range of k values. Output fit quality per k.

```bash
narcissus optimize \
  --data streamflow.csv \
  --min-k 4 \
  --max-k 20 \
  --experiment my_run \
  --output-dir results/
```

```json
{
  "experiment": "my_run",
  "n_basins": 24000,
  "results": [
    {"k": 4, "inertia": 2834.2},
    {"k": 5, "inertia": 2567.1}
  ]
}
```

### `narcissus cluster`

Cluster for a single k. Save assignments and centroids.

```bash
narcissus cluster \
  --data streamflow.csv \
  --k 15 \
  --experiment my_run \
  --output-dir results/
```

```json
{
  "experiment": "my_run",
  "k": 15,
  "inertia": 1523.4,
  "n_basins": 24000,
  "cluster_sizes": [1200, 980, 1450]
}
```

### `narcissus evaluate`

Train a classifier on cluster labels + basin attributes. Report cross-validated accuracy.

```bash
narcissus evaluate \
  --experiment my_run \
  --k 15 \
  --attributes basin_attrs.csv \
  --cv-folds 5 \
  --output-dir results/
```

```json
{
  "experiment": "my_run",
  "k": 15,
  "cv_accuracy": 0.856,
  "n_features": 12,
  "top_features": [
    {"attribute": "frac_snow", "importance": 0.23, "rank": 1},
    {"attribute": "elevation_m", "importance": 0.18, "rank": 2}
  ]
}
```

### `narcissus predict`

Predict cluster membership for new basins using a trained model.

```bash
narcissus predict \
  --experiment my_run \
  --k 15 \
  --attributes new_basins.csv \
  --top-k 3 \
  --output-dir results/
```

```json
{
  "experiment": "my_run",
  "k": 15,
  "n_predicted": 500,
  "predictions": [
    {
      "basin_id": "NEW_001",
      "clusters": [
        {"cluster": 5, "probability": 0.72},
        {"cluster": 12, "probability": 0.15},
        {"cluster": 3, "probability": 0.08}
      ]
    }
  ]
}
```

### Common Flags

| Flag | Default | Description |
|---|---|---|
| `--seed <u64>` | `42` | RNG seed for reproducibility |
| `--threads <usize>` | all cores | Rayon thread pool size |
| `--verbose` | off | Debug-level tracing to stderr |
| `--quiet` | off | Suppress all stderr except errors |

### Clustering Tuning Flags

| Flag | Default | Description |
|---|---|---|
| `--n-init <usize>` | `10` | K-means restarts (keep best) |
| `--max-iter <usize>` | `75` | Max iterations per run |
| `--warping-window <usize>` | `2` | Sakoe-Chiba radius (time steps) |
| `--tol <f64>` | `1e-4` | Convergence tolerance |

---

## Performance Strategy

- **DTW inner loop**: Cache-friendly DP matrix layout, potential SIMD, reusable scratch buffers
- **Parallelism**: Rayon for distance matrix computation, tree building, cross-validation folds
- **Memory**: Arena/pool allocation for DP matrices across basin pairs; contiguous row-major time series storage
- **I/O**: Memory-mapped files for large datasets where possible
