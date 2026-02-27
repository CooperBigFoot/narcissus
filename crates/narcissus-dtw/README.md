# narcissus-dtw

Pure math library for Dynamic Time Warping distance computation and DBA barycenter averaging. Zero I/O — consumed by `narcissus-cluster` for K-means iteration.

## Architecture

```mermaid
flowchart TD
    subgraph Input["Input / Validation"]
        series["series.rs\nTimeSeries · TimeSeriesView"]
        constraint["constraint.rs\nBandConstraint"]
        error["error.rs\nDtwError · DbaError"]
    end

    subgraph Core["Core Computation"]
        dtw["dtw.rs\nDtw\ndistance() · distance_and_path() · pairwise()"]
        dba["dba.rs\nDbaConfig · DbaResult\naverage()"]
    end

    subgraph Output["Output Types"]
        distance["distance.rs\nDtwDistance"]
        path["path.rs\nWarpingPath · WarpingStep"]
        matrix["matrix.rs\nDistanceMatrix"]
    end

    series -->|TimeSeriesView| dtw
    series -->|TimeSeriesView| dba
    constraint -->|BandConstraint| dtw
    constraint -->|BandConstraint| dba
    error -->|DtwError| series
    error -->|DbaError| dba

    dtw -->|DtwDistance| distance
    dtw -->|WarpingPath| path
    dtw -->|DistanceMatrix via pairwise()| matrix
    dtw -->|distance_and_path()| dba

    distance --> matrix
```

Module dependency flow: series and constraint are consumed by the two computation modules (dtw and dba). dtw produces all three output types and is also called internally by dba for per-series alignment during each barycenter iteration.

## Glossary

| Term | Definition |
|---|---|
| DTW | Dynamic Time Warping — elastic distance metric that aligns time series by warping the time axis |
| DBA | DTW Barycenter Averaging — iterative algorithm to compute a centroid under DTW alignment (Petitjean et al. 2011) |
| Sakoe-Chiba band | Constraint limiting warping: cell (i,j) is valid only if \|i-j\| ≤ r. Reduces complexity from O(n²) to O(n·r) |
| Warping path | Sequence of index pairs (i,j) defining the optimal alignment between two series |
| Cost matrix | DP matrix where C[i][j] = squared Euclidean cost + min of three predecessors (diagonal, above, left) |
| Band width | Number of valid columns per row under the Sakoe-Chiba constraint: min(2r+1, m) |
| Barycenter | The centroid time series minimizing total DTW distance to all input series |
| Rolling buffer | Two-row sliding window used by `distance()` to avoid allocating the full cost matrix |
| Traceback | Backward pass through stored direction bits to reconstruct the optimal warping path |

## Key Types

| Type | Module | Role |
|---|---|---|
| `Dtw` | `dtw.rs` | Immutable config (constraint). Entry point for `distance()`, `distance_and_path()`, `pairwise()` |
| `TimeSeries` | `series.rs` | Owned, validated time series (non-empty, all values finite) |
| `TimeSeriesView` | `series.rs` | Borrowed zero-copy view into a validated series |
| `DtwDistance` | `distance.rs` | Newtype for non-negative distance values; provides `total_cmp` for ordering |
| `BandConstraint` | `constraint.rs` | `Unconstrained` or `SakoeChibaRadius(r)` — controls the warping window |
| `WarpingPath` | `path.rs` | Ordered sequence of `WarpingStep { a, b }` from `(0,0)` to `(n-1,m-1)` |
| `DistanceMatrix` | `matrix.rs` | Lower-triangular symmetric matrix produced by `Dtw::pairwise()` |
| `DbaConfig` | `dba.rs` | Builder config: constraint, `max_iter`, `tol` |
| `DbaResult` | `dba.rs` | Output of `DbaConfig::average()`: centroid, convergence metadata |
| `DtwError` | `error.rs` | Validation errors: `EmptySeries`, `NonFiniteValue` |
| `DbaError` | `error.rs` | DBA errors: `EmptyCluster`, `Dtw(DtwError)` |

## Performance Notes

- **Distance-only fast path**: `Dtw::distance()` uses a rolling two-row buffer. For n=52, r=2: 112 bytes of working memory.
- **Full path**: `Dtw::distance_and_path()` allocates a band-compressed cost matrix. For n=52, r=2: 2.08 KB (vs. 21.6 KB for the full unconstrained matrix).
- **Pairwise**: `Dtw::pairwise()` is Rayon-parallelized across all n(n-1)/2 pairs.
- **DBA**: Each iteration calls `distance_and_path()` once per input series. Convergence is checked against a configurable tolerance (`tol = 1e-5` default).
