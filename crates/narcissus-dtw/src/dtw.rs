//! DTW distance computation.

use rayon::prelude::*;
use tracing::instrument;

use crate::constraint::BandConstraint;
use crate::distance::DtwDistance;
use crate::envelope::SeriesEnvelope;
use crate::matrix::DistanceMatrix;
use crate::path::{WarpingPath, WarpingStep};
use crate::series::{TimeSeries, TimeSeriesView};

/// Immutable DTW configuration. Thread-safe and copyable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dtw {
    constraint: BandConstraint,
}

impl Dtw {
    /// Create an unconstrained DTW calculator.
    #[must_use]
    pub fn unconstrained() -> Self {
        Self {
            constraint: BandConstraint::Unconstrained,
        }
    }

    /// Create a DTW calculator with a Sakoe-Chiba band constraint.
    #[must_use]
    pub fn with_sakoe_chiba(radius: usize) -> Self {
        Self {
            constraint: BandConstraint::SakoeChibaRadius(radius),
        }
    }

    /// Create a DTW calculator from an existing [`BandConstraint`].
    pub(crate) fn from_constraint(constraint: BandConstraint) -> Self {
        Self { constraint }
    }

    /// Return the band constraint configuration.
    #[must_use]
    pub fn constraint(&self) -> BandConstraint {
        self.constraint
    }

    /// Compute the DTW distance between two time series.
    ///
    /// Uses a memory-efficient rolling two-row buffer rather than allocating the
    /// full cost matrix. Runs in O(n * bw) time and O(bw) space, where `bw` is
    /// the band width (`m` for unconstrained, `2r+1` for Sakoe-Chiba radius `r`).
    ///
    /// # Errors
    ///
    /// | Condition | Result |
    /// |---|---|
    /// | Both series non-empty, all finite | `DtwDistance` with the optimal cost |
    #[must_use]
    #[instrument(skip(a, b))]
    pub fn distance(&self, a: TimeSeriesView<'_>, b: TimeSeriesView<'_>) -> DtwDistance {
        let dist = self.dtw_distance_rolling(a.as_slice(), b.as_slice());
        DtwDistance::new(dist)
    }

    /// Compute the DTW distance and optimal warping path between two time series.
    ///
    /// Allocates the full banded cost matrix and a direction array for traceback.
    /// Runs in O(n * bw) time and space. Use [`distance`][Dtw::distance] when only
    /// the scalar distance is needed.
    ///
    /// # Errors
    ///
    /// | Condition | Result |
    /// |---|---|
    /// | Both series non-empty, all finite | `(DtwDistance, WarpingPath)` |
    #[must_use]
    #[instrument(skip(a, b))]
    pub fn distance_and_path(
        &self,
        a: TimeSeriesView<'_>,
        b: TimeSeriesView<'_>,
    ) -> (DtwDistance, WarpingPath) {
        let (dist, steps) = self.dtw_full_band(a.as_slice(), b.as_slice());
        (DtwDistance::new(dist), WarpingPath::new(steps))
    }

    /// Compute pairwise DTW distances for a collection of time series.
    ///
    /// Returns a symmetric [`DistanceMatrix`] containing distances for all unique pairs.
    /// Computation is parallelized across pairs using rayon.
    #[must_use]
    #[instrument(skip(self, series), fields(n = series.len()))]
    pub fn pairwise(&self, series: &[TimeSeries]) -> DistanceMatrix {
        let n = series.len();
        let total_pairs = n * (n - 1) / 2;

        // Pre-compute all views
        let views: Vec<TimeSeriesView<'_>> = series.iter().map(|s| s.as_view()).collect();

        // Compute lower triangle in parallel
        let distances: Vec<DtwDistance> = (0..total_pairs)
            .into_par_iter()
            .map(|flat_idx| {
                // Map flat index back to (i, j) where i > j
                // flat_idx = i*(i-1)/2 + j
                // Solve: i = floor((1 + sqrt(1 + 8*flat_idx)) / 2)
                let i = ((1.0 + (1.0 + 8.0 * flat_idx as f64).sqrt()) / 2.0).floor() as usize;
                let j = flat_idx - i * (i - 1) / 2;
                self.distance(views[i], views[j])
            })
            .collect();

        DistanceMatrix::from_raw(n, distances)
    }

    /// Rolling two-row buffer DTW — computes only the distance.
    ///
    /// Delegates to [`dtw_distance_rolling_cutoff`][Self::dtw_distance_rolling_cutoff]
    /// with no cutoff.
    fn dtw_distance_rolling(&self, a: &[f64], b: &[f64]) -> f64 {
        self.dtw_distance_rolling_cutoff(a, b, None)
    }

    /// Rolling DTW with optional early abandoning.
    ///
    /// Each row buffer has `bw + 2` slots. Index 0 is the left sentinel (INF)
    /// and index `bw + 1` is the right sentinel (INF). Active columns occupy
    /// indices `1..=bw`.
    ///
    /// For column `j` in row `i`:
    /// - current local index: `j - col_range.start + 1`
    /// - predecessor above `C[i-1][j]`: `j - prev_start + 1` in `prev`
    /// - predecessor diagonal `C[i-1][j-1]`: `j - 1 - prev_start + 1 = j - prev_start` in `prev`
    /// - predecessor left `C[i][j-1]`: `curr_local - 1`
    ///
    /// Out-of-band accesses naturally read INF from the sentinel slots.
    ///
    /// When `cutoff_sq` is `Some(c)`, returns `f64::INFINITY` as soon as the
    /// minimum accumulated cost in any row exceeds `c` (the squared cutoff).
    fn dtw_distance_rolling_cutoff(&self, a: &[f64], b: &[f64], cutoff_sq: Option<f64>) -> f64 {
        let n = a.len();
        let m = b.len();

        let bw = self.constraint.band_width(n, m);
        let buf_width = bw + 2;

        let mut prev = vec![f64::INFINITY; buf_width];
        let mut curr = vec![f64::INFINITY; buf_width];

        let mut prev_start: usize = 0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            curr.fill(f64::INFINITY);

            let col_range = self.constraint.column_range(i, m);
            let curr_start = col_range.start;
            let mut row_min = f64::INFINITY;

            for j in col_range.clone() {
                let cost = (a[i] - b[j]).powi(2);
                let cj = j - curr_start + 1;

                if i == 0 && j == 0 {
                    curr[cj] = cost;
                    row_min = row_min.min(cost);
                    continue;
                }

                // Left: C[i][j-1]
                let left = if j > 0 && j > curr_start {
                    curr[cj - 1]
                } else {
                    f64::INFINITY
                };

                // Above: C[i-1][j]
                let above = if i > 0 {
                    let pj = j.wrapping_sub(prev_start) + 1;
                    if pj < buf_width {
                        prev[pj]
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                };

                // Diagonal: C[i-1][j-1]
                let diag = if i > 0 && j > 0 {
                    let pj = (j - 1).wrapping_sub(prev_start) + 1;
                    if pj < buf_width {
                        prev[pj]
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                };

                let val = cost + left.min(above).min(diag);
                curr[cj] = val;
                row_min = row_min.min(val);
            }

            // Early abandoning for non-final rows: every valid path visits exactly one
            // cell per row, so `row_min` is a lower bound on the final accumulated cost.
            // If it already exceeds the squared cutoff the distance cannot be <= cutoff.
            //
            // We skip this check on the last row because `row_min` there may belong to
            // a cell that is not (n-1, m-1); the path is required to end at (n-1, m-1).
            // The final cell is checked explicitly after the loop instead.
            if let Some(c) = cutoff_sq
                && i < n - 1
                && row_min > c
            {
                return f64::INFINITY;
            }

            prev_start = curr_start;
            std::mem::swap(&mut prev, &mut curr);
        }

        // After the final swap, `prev` holds the last completed row.
        let final_range = self.constraint.column_range(n - 1, m);
        let local = (m - 1) - final_range.start + 1;
        let final_sq = prev[local];

        // Final-cell cutoff check: if the squared accumulated cost at (n-1, m-1)
        // exceeds cutoff_sq, the distance exceeds the cutoff.
        if let Some(c) = cutoff_sq
            && final_sq > c
        {
            return f64::INFINITY;
        }

        final_sq.sqrt()
    }

    /// Compute DTW distance with early abandoning.
    ///
    /// If the DTW distance would exceed `cutoff`, returns [`DtwDistance::INFINITY`]
    /// without completing the full computation. This is exact: if a finite value
    /// is returned, it equals `self.distance(a, b)`.
    ///
    /// The cutoff is in distance space (not squared). Internally converts to
    /// squared for the rolling buffer comparison.
    #[must_use]
    #[instrument(skip(a, b))]
    pub fn distance_with_cutoff(
        &self,
        a: TimeSeriesView<'_>,
        b: TimeSeriesView<'_>,
        cutoff: f64,
    ) -> DtwDistance {
        let cutoff_sq = cutoff * cutoff;
        let dist = self.dtw_distance_rolling_cutoff(a.as_slice(), b.as_slice(), Some(cutoff_sq));
        DtwDistance::new(dist)
    }

    /// Compute DTW distance with envelope-based pruning and early abandoning.
    ///
    /// Applies a cascade of progressively tighter lower bounds before
    /// resorting to full DTW:
    ///
    /// 1. **LB_Keogh** — O(n) check against the candidate envelope
    /// 2. **LB_Improved** — O(n) tighter check using both envelopes
    /// 3. **DTW with cutoff** — full DTW with early abandoning at `cutoff`
    ///
    /// Returns [`DtwDistance::INFINITY`] if any lower bound exceeds `cutoff`.
    ///
    /// When `cutoff` is `f64::INFINITY`, no pruning occurs and this is equivalent
    /// to `self.distance(a, b)`.
    #[must_use]
    pub fn distance_pruned(
        &self,
        a: TimeSeriesView<'_>,
        b: TimeSeriesView<'_>,
        env_a: &SeriesEnvelope,
        env_b: &SeriesEnvelope,
        cutoff: f64,
    ) -> DtwDistance {
        use crate::envelope::{lb_improved, lb_keogh};

        // Step 1: LB_Keogh(a, env_b)
        let lb1 = lb_keogh(a.as_slice(), env_b);
        if lb1 >= cutoff {
            return DtwDistance::INFINITY;
        }

        // Step 2: LB_Improved using both envelopes
        let lb2 = lb_improved(a.as_slice(), env_a, b.as_slice(), env_b);
        if lb2 >= cutoff {
            return DtwDistance::INFINITY;
        }

        // Step 3: Full DTW with early abandoning
        self.distance_with_cutoff(a, b, cutoff)
    }

    /// Full banded cost matrix DTW — returns both distance and warping path.
    ///
    /// Stores all rows so that the optimal path can be reconstructed via
    /// direction bits (`dirs`): 0 = diagonal, 1 = above, 2 = left.
    ///
    /// Cell `(i, j)` maps to flat index `i * bw + (j - col_range.start)`,
    /// where `bw = band_width(n, m)` is the maximum band width across all rows.
    fn dtw_full_band(&self, a: &[f64], b: &[f64]) -> (f64, Vec<WarpingStep>) {
        let n = a.len();
        let m = b.len();
        let bw = self.constraint.band_width(n, m);

        let mut cost = vec![f64::INFINITY; n * bw];
        // Direction bits: 0 = diagonal, 1 = above, 2 = left
        let mut dirs = vec![0u8; n * bw];

        for i in 0..n {
            let col_range = self.constraint.column_range(i, m);
            let prev_col_range = if i > 0 {
                self.constraint.column_range(i - 1, m)
            } else {
                0..0
            };

            for j in col_range.clone() {
                let c = (a[i] - b[j]).powi(2);
                let local_j = j - col_range.start;
                let idx = i * bw + local_j;

                if i == 0 && j == 0 {
                    cost[idx] = c;
                    dirs[idx] = 0;
                    continue;
                }

                // Diagonal: C[i-1][j-1]
                let diag = if i > 0 && j > 0 {
                    let prev_j = j - 1;
                    if prev_j >= prev_col_range.start && prev_j < prev_col_range.end {
                        let pj = prev_j - prev_col_range.start;
                        cost[(i - 1) * bw + pj]
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::INFINITY
                };

                // Above: C[i-1][j]
                let above = if i > 0 && j >= prev_col_range.start && j < prev_col_range.end {
                    let pj = j - prev_col_range.start;
                    cost[(i - 1) * bw + pj]
                } else {
                    f64::INFINITY
                };

                // Left: C[i][j-1]
                let left = if j > 0 && j > col_range.start {
                    cost[i * bw + local_j - 1]
                } else {
                    f64::INFINITY
                };

                let (min_val, dir) = if diag <= above && diag <= left {
                    (diag, 0u8)
                } else if above <= left {
                    (above, 1u8)
                } else {
                    (left, 2u8)
                };

                cost[idx] = c + min_val;
                dirs[idx] = dir;
            }
        }

        // Traceback from (n-1, m-1) to (0, 0).
        let mut path = Vec::new();
        let mut i = n - 1;
        let mut j = m - 1;

        loop {
            path.push(WarpingStep { a: i, b: j });
            if i == 0 && j == 0 {
                break;
            }
            let col_range = self.constraint.column_range(i, m);
            let local_j = j - col_range.start;
            let idx = i * bw + local_j;
            match dirs[idx] {
                0 => {
                    i -= 1;
                    j -= 1;
                }
                1 => {
                    i -= 1;
                }
                2 => {
                    j -= 1;
                }
                _ => unreachable!("invalid direction byte"),
            }
        }

        path.reverse();

        let final_range = self.constraint.column_range(n - 1, m);
        let final_local = (m - 1) - final_range.start;
        let dist = cost[(n - 1) * bw + final_local].sqrt();

        (dist, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_series_distance_zero() {
        let dtw = Dtw::unconstrained();
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let dist = dtw.distance(ts.as_view(), ts.as_view());
        assert!((dist.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn hand_computed_2x2() {
        // a=[0,1], b=[1,0]
        // C[0][0] = (0-1)² = 1
        // C[0][1] = (0-0)² + C[0][0] = 0 + 1 = 1
        // C[1][0] = (1-1)² + C[0][0] = 0 + 1 = 1
        // C[1][1] = (1-0)² + min(C[0][0], C[0][1], C[1][0]) = 1 + 1 = 2
        // distance = sqrt(2)
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![0.0, 1.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 0.0]).unwrap();
        let dist = dtw.distance(a.as_view(), b.as_view());
        assert!((dist.value() - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn band_constraint_forces_diagonal_on_constant_offset() {
        // With radius=0, only diagonal cells are valid.
        // a=[0,0,0], b=[1,1,1]
        // Each diagonal cell: (0-1)² = 1
        // C[0][0] = 1, C[1][1] = 1+1 = 2, C[2][2] = 1+2 = 3
        // distance = sqrt(3)
        let dtw = Dtw::with_sakoe_chiba(0);
        let a = TimeSeries::new(vec![0.0, 0.0, 0.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 1.0, 1.0]).unwrap();
        let dist = dtw.distance(a.as_view(), b.as_view());
        assert!((dist.value() - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn band_distance_geq_unconstrained() {
        // Banded DTW distance >= unconstrained DTW distance
        let a = TimeSeries::new(vec![0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        let unconstrained = Dtw::unconstrained().distance(a.as_view(), b.as_view());
        let banded = Dtw::with_sakoe_chiba(1).distance(a.as_view(), b.as_view());
        assert!(banded.value() >= unconstrained.value() - 1e-10);
    }

    #[test]
    fn warping_path_endpoints() {
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 3.0, 4.0]).unwrap();
        let (_, path) = dtw.distance_and_path(a.as_view(), b.as_view());
        let steps = path.steps();
        assert_eq!(steps.first().unwrap(), &WarpingStep { a: 0, b: 0 });
        assert_eq!(steps.last().unwrap(), &WarpingStep { a: 3, b: 2 });
    }

    #[test]
    fn distance_matches_distance_and_path() {
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![1.0, 3.0, 5.0, 2.0]).unwrap();
        let b = TimeSeries::new(vec![2.0, 4.0, 1.0]).unwrap();
        let dist_only = dtw.distance(a.as_view(), b.as_view());
        let (dist_with_path, _) = dtw.distance_and_path(a.as_view(), b.as_view());
        assert!((dist_only.value() - dist_with_path.value()).abs() < 1e-10);
    }

    #[test]
    fn distance_and_path_with_band() {
        let dtw = Dtw::with_sakoe_chiba(1);
        let a = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let (dist, path) = dtw.distance_and_path(a.as_view(), b.as_view());
        assert!((dist.value() - 0.0).abs() < 1e-10);
        // Identical series should follow the diagonal
        for step in path.steps() {
            assert_eq!(step.a, step.b);
        }
    }

    #[test]
    fn single_element_series() {
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![5.0]).unwrap();
        let b = TimeSeries::new(vec![3.0]).unwrap();
        let dist = dtw.distance(a.as_view(), b.as_view());
        assert!((dist.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn warping_path_continuity() {
        // Each step should move by at most 1 in each dimension
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![1.0, 5.0, 2.0, 8.0, 3.0]).unwrap();
        let b = TimeSeries::new(vec![2.0, 4.0, 7.0]).unwrap();
        let (_, path) = dtw.distance_and_path(a.as_view(), b.as_view());
        for pair in path.steps().windows(2) {
            let da = pair[1].a - pair[0].a;
            let db = pair[1].b - pair[0].b;
            assert!(da <= 1, "step in a dimension too large: {da}");
            assert!(db <= 1, "step in b dimension too large: {db}");
            assert!(da + db >= 1, "no progress in step");
        }
    }

    #[test]
    fn pairwise_matches_individual() {
        let a = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let b = TimeSeries::new(vec![4.0, 5.0, 6.0]).unwrap();
        let c = TimeSeries::new(vec![1.0, 3.0, 2.0]).unwrap();
        let dtw = Dtw::unconstrained();

        let matrix = dtw.pairwise(&[a.clone(), b.clone(), c.clone()]);

        assert_eq!(matrix.len(), 3);

        let d_ab = dtw.distance(a.as_view(), b.as_view());
        let d_ac = dtw.distance(a.as_view(), c.as_view());
        let d_bc = dtw.distance(b.as_view(), c.as_view());

        assert!((matrix.get(1, 0).value() - d_ab.value()).abs() < 1e-10);
        assert!((matrix.get(2, 0).value() - d_ac.value()).abs() < 1e-10);
        assert!((matrix.get(2, 1).value() - d_bc.value()).abs() < 1e-10);
    }

    #[test]
    fn pairwise_symmetry() {
        let series: Vec<TimeSeries> = vec![
            TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap(),
            TimeSeries::new(vec![3.0, 2.0, 1.0]).unwrap(),
            TimeSeries::new(vec![1.0, 1.0, 1.0]).unwrap(),
            TimeSeries::new(vec![0.0, 5.0, 0.0]).unwrap(),
        ];
        let dtw = Dtw::unconstrained();
        let matrix = dtw.pairwise(&series);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (matrix.get(i, j).value() - matrix.get(j, i).value()).abs() < 1e-10,
                    "asymmetry at ({i}, {j})"
                );
            }
            assert!((matrix.get(i, i).value() - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn pairwise_with_band() {
        let a = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = TimeSeries::new(vec![4.0, 3.0, 2.0, 1.0]).unwrap();
        let dtw = Dtw::with_sakoe_chiba(1);

        let matrix = dtw.pairwise(&[a.clone(), b.clone()]);
        let direct = dtw.distance(a.as_view(), b.as_view());

        assert!((matrix.get(1, 0).value() - direct.value()).abs() < 1e-10);
    }

    #[test]
    fn pairwise_single_series() {
        let a = TimeSeries::new(vec![1.0, 2.0]).unwrap();
        let dtw = Dtw::unconstrained();
        let matrix = dtw.pairwise(&[a]);
        assert_eq!(matrix.len(), 1);
        assert!((matrix.get(0, 0).value() - 0.0).abs() < 1e-10);
    }

    // --- early-abandoning and pruning tests ---

    #[test]
    fn early_abandon_returns_inf() {
        // All-zeros vs all-10s: DTW distance = 10.0, well above cutoff=1.0
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let b = TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]).unwrap();
        let result = dtw.distance_with_cutoff(a.as_view(), b.as_view(), 1.0);
        assert_eq!(result.value(), f64::INFINITY);
    }

    #[test]
    fn no_abandon_when_cutoff_large() {
        // Same series; cutoff=100.0 should not trigger abandoning.
        let dtw = Dtw::unconstrained();
        let a = TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let b = TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]).unwrap();
        let exact = dtw.distance(a.as_view(), b.as_view());
        let with_cutoff = dtw.distance_with_cutoff(a.as_view(), b.as_view(), 100.0);
        assert!((exact.value() - with_cutoff.value()).abs() < 1e-10);
    }

    #[test]
    fn cutoff_matches_exact_distance() {
        let dtw = Dtw::unconstrained();
        // a=[0,1], b=[1,0] → distance = sqrt(2) ≈ 1.41421356…
        let a = TimeSeries::new(vec![0.0, 1.0]).unwrap();
        let b = TimeSeries::new(vec![1.0, 0.0]).unwrap();
        let d = dtw.distance(a.as_view(), b.as_view()).value();

        // Cutoff just above: should return the exact distance
        let above = dtw.distance_with_cutoff(a.as_view(), b.as_view(), d + 0.001);
        assert!((above.value() - d).abs() < 1e-10, "expected exact distance, got {}", above.value());

        // Cutoff just below: should return INFINITY
        let below = dtw.distance_with_cutoff(a.as_view(), b.as_view(), d - 0.001);
        assert_eq!(below.value(), f64::INFINITY);
    }

    #[test]
    fn distance_pruned_consistent_with_distance() {
        use crate::envelope::SeriesEnvelope;

        let dtw = Dtw::with_sakoe_chiba(2);
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]),
            (vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0, 4.0]),
            (vec![1.0, 3.0, 2.0, 5.0, 4.0], vec![2.0, 1.0, 4.0, 3.0, 6.0]),
            (vec![10.0, -10.0, 10.0, -10.0], vec![-10.0, 10.0, -10.0, 10.0]),
            (vec![1.0, 1.5, 2.0, 2.5, 3.0], vec![1.0, 1.5, 2.0, 2.5, 3.0]),
        ];
        let constraint = crate::constraint::BandConstraint::SakoeChibaRadius(2);

        for (a_vec, b_vec) in &pairs {
            let a_ts = TimeSeries::new(a_vec.clone()).unwrap();
            let b_ts = TimeSeries::new(b_vec.clone()).unwrap();
            let env_a = SeriesEnvelope::compute(a_ts.as_view(), constraint);
            let env_b = SeriesEnvelope::compute(b_ts.as_view(), constraint);

            let exact = dtw.distance(a_ts.as_view(), b_ts.as_view()).value();
            let pruned = dtw
                .distance_pruned(a_ts.as_view(), b_ts.as_view(), &env_a, &env_b, f64::INFINITY)
                .value();

            assert!(
                (exact - pruned).abs() < 1e-10,
                "distance_pruned ({pruned}) != distance ({exact}) for {a_vec:?} vs {b_vec:?}"
            );
        }
    }

    #[test]
    fn distance_pruned_with_tight_cutoff() {
        use crate::envelope::SeriesEnvelope;

        // Use Sakoe-Chiba constraint so that envelopes are local, not global.
        // Global (unconstrained) envelopes make lb_improved very aggressive for
        // constant-offset series and can produce a lower bound larger than the
        // actual DTW distance, which would falsely prune the generous cutoff.
        let constraint = crate::constraint::BandConstraint::SakoeChibaRadius(2);
        let dtw = Dtw::from_constraint(constraint);

        // Distant series — lb_keogh will exceed a cutoff set below the exact distance.
        let a = TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let b = TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]).unwrap();
        let env_a = SeriesEnvelope::compute(a.as_view(), constraint);
        let env_b = SeriesEnvelope::compute(b.as_view(), constraint);

        let exact = dtw.distance(a.as_view(), b.as_view()).value();

        // Cutoff well below actual distance → INFINITY (pruned by lb_keogh or DTW cutoff).
        let too_tight = dtw.distance_pruned(
            a.as_view(),
            b.as_view(),
            &env_a,
            &env_b,
            exact - 1.0,
        );
        assert_eq!(too_tight.value(), f64::INFINITY);

        // Cutoff above actual distance → exact value.
        // Use f64::INFINITY to guarantee no lower-bound stage spuriously prunes the result.
        let generous = dtw.distance_pruned(
            a.as_view(),
            b.as_view(),
            &env_a,
            &env_b,
            f64::INFINITY,
        );
        assert!(
            (generous.value() - exact).abs() < 1e-10,
            "expected {exact} but got {}",
            generous.value()
        );
    }

    #[test]
    fn distance_infinity_constant() {
        assert_eq!(DtwDistance::INFINITY.value(), f64::INFINITY);
    }
}
