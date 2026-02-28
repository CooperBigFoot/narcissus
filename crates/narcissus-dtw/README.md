# narcissus-dtw

Pure math library for Dynamic Time Warping distance computation, DBA/SSG barycenter averaging, and time series preprocessing. Zero I/O — consumed by `narcissus-cluster` for K-means iteration.

## Architecture

```mermaid
flowchart TD
    subgraph Input["Input / Validation"]
        series["series.rs\nTimeSeries · TimeSeriesView"]
        constraint["constraint.rs\nBandConstraint"]
        error["error.rs\nDtwError · DbaError\nPreprocessError · DerivativeError"]
    end

    subgraph Preprocess["Preprocessing"]
        preprocess["preprocess.rs\nz_normalize() · z_normalize_batch()\nderivative()"]
    end

    subgraph Pruning["Envelope / Pruning"]
        envelope["envelope.rs\nSeriesEnvelope\nlb_keogh() · lb_improved()"]
    end

    subgraph Core["Core Computation"]
        dtw["dtw.rs\nDtw\ndistance() · distance_and_path()\npairwise() · distance_pruned()"]
        dba["dba.rs\nDbaConfig · DbaResult\nDbaInit · DbaMode\naverage()"]
        ssg["ssg.rs\nSsgConfig · SsgResult\nStochastic Subgradient averaging"]
    end

    subgraph Output["Output Types"]
        distance["distance.rs\nDtwDistance"]
        path["path.rs\nWarpingPath · WarpingStep"]
        matrix["matrix.rs\nDistanceMatrix"]
    end

    series -->|TimeSeriesView| dtw
    series -->|TimeSeriesView| dba
    series -->|TimeSeriesView| ssg
    constraint -->|BandConstraint| dtw
    constraint -->|BandConstraint| dba
    constraint -->|BandConstraint| ssg
    constraint -->|BandConstraint| envelope
    error -->|DtwError| series
    error -->|DbaError| dba
    error -->|PreprocessError| preprocess
    error -->|DerivativeError| preprocess

    envelope -->|SeriesEnvelope| dtw
    dtw -->|DtwDistance| distance
    dtw -->|WarpingPath| path
    dtw -->|DistanceMatrix via pairwise()| matrix
    dtw -->|distance_and_path()| dba
    dtw -->|distance_and_path()| ssg

    distance --> matrix
```

Module dependency flow: series and constraint are consumed by the three computation modules (dtw, dba, ssg). The envelope module precomputes upper/lower envelopes under a Sakoe-Chiba band and feeds into dtw for cascaded pruning via `distance_pruned()`. The preprocess module is standalone — consumed externally for z-normalization and derivative transforms. dtw produces all three output types and is called internally by both dba and ssg for per-series alignment during each iteration.

## Glossary

| Term | Definition |
|---|---|
| DTW | Dynamic Time Warping — elastic distance metric that aligns time series by warping the time axis |
| DBA | DTW Barycenter Averaging — iterative algorithm to compute a centroid under DTW alignment (Petitjean et al. 2011) |
| SSG | Stochastic Subgradient averaging — online centroid update with decaying learning rate |
| Sakoe-Chiba band | Constraint limiting warping: cell (i,j) is valid only if \|i-j\| ≤ r. Reduces complexity from O(n²) to O(n·r) |
| LB_Keogh | O(n) lower bound on DTW distance using precomputed upper/lower envelopes |
| LB_Improved | Tighter O(n) lower bound using envelopes of both query and candidate series |
| Early abandoning | Stop DTW computation early when running sum exceeds a cutoff threshold |
| PrunedDTW | Dynamic column pruning that tightens the Sakoe-Chiba band beyond the static constraint |
| Z-normalization | Transform to zero mean, unit variance — standard preprocessing for shape-based DTW |
| Derivative DTW | DTW on first derivatives rather than raw values; captures shape changes |
| Warping path | Sequence of index pairs (i,j) defining the optimal alignment between two series |
| Cost matrix | DP matrix where C[i][j] = squared Euclidean cost + min of three predecessors (diagonal, above, left) |
| Band width | Number of valid columns per row under the Sakoe-Chiba constraint: min(2r+1, m) |
| Barycenter | The centroid time series minimizing total DTW distance to all input series |
| Medoid init | Initialize DBA centroid as the input series minimizing total DTW distance to all others |
| Stochastic DBA | DBA with random subset sampling per iteration; trades accuracy for speed on large clusters |
| Rolling buffer | Two-row sliding window used by `distance()` to avoid allocating the full cost matrix |
| Traceback | Backward pass through stored direction bits to reconstruct the optimal warping path |

## Key Types

| Type | Module | Role |
|---|---|---|
| `Dtw` | `dtw.rs` | Immutable config (constraint). Entry point for `distance()`, `distance_and_path()`, `pairwise()`, `distance_pruned()` |
| `TimeSeries` | `series.rs` | Owned, validated time series (non-empty, all values finite) |
| `TimeSeriesView` | `series.rs` | Borrowed zero-copy view into a validated series |
| `DtwDistance` | `distance.rs` | Newtype for non-negative distance values; provides `total_cmp` for ordering. `INFINITY` sentinel for pruned-out pairs |
| `BandConstraint` | `constraint.rs` | `Unconstrained` or `SakoeChibaRadius(r)` — controls the warping window |
| `WarpingPath` | `path.rs` | Ordered sequence of `WarpingStep { a, b }` from `(0,0)` to `(n-1,m-1)` |
| `DistanceMatrix` | `matrix.rs` | Lower-triangular symmetric matrix produced by `Dtw::pairwise()` |
| `SeriesEnvelope` | `envelope.rs` | Precomputed upper/lower envelopes for a series under Sakoe-Chiba constraint |
| `DbaConfig` | `dba.rs` | Builder config: constraint, `max_iter`, `tol`, `init`, `sample_fraction`, `seed` |
| `DbaResult` | `dba.rs` | Output of `DbaConfig::average()`: centroid, convergence metadata |
| `DbaInit` | `dba.rs` | Enum: `ElementWiseMean` or `Medoid` — DBA initialization strategy |
| `DbaMode` | `dba.rs` | Enum: `Full` or `Stochastic` — DBA mode indicator in results |
| `SsgConfig` | `ssg.rs` | Builder config for SSG averaging: constraint, max_epochs, lr_init, decay, tol, seed |
| `SsgResult` | `ssg.rs` | Output of SSG averaging: centroid, converged flag, epoch count |
| `DtwError` | `error.rs` | Validation errors: `EmptySeries`, `NonFiniteValue` |
| `DbaError` | `error.rs` | DBA errors: `EmptyCluster`, `Dtw(DtwError)` |
| `PreprocessError` | `error.rs` | Z-normalization errors: `ConstantSeries` |
| `DerivativeError` | `error.rs` | Derivative errors: `TooShort { len, min: 3 }` |

## Performance Notes

- **Distance-only fast path**: `Dtw::distance()` uses a rolling two-row buffer. For n=52, r=2: 112 bytes of working memory.
- **Full path**: `Dtw::distance_and_path()` allocates a band-compressed cost matrix. For n=52, r=2: 2.08 KB (vs. 21.6 KB for the full unconstrained matrix).
- **Pairwise**: `Dtw::pairwise()` is Rayon-parallelized across all n(n-1)/2 pairs.
- **DBA**: Each iteration calls `distance_and_path()` once per input series. Convergence is checked against a configurable tolerance (`tol = 1e-5` default).
- **LB_Keogh**: O(n) lower bound prunes ~80% of full DTW computations in typical K-means scenarios.
- **Early abandoning**: Returns `INFINITY` as soon as running cost exceeds cutoff, skipping remaining cells.
- **PrunedDTW**: Dynamically narrows the Sakoe-Chiba band by tracking active column ranges per row.
