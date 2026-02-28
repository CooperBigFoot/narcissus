//! Elkan's triangle inequality bounds for accelerated K-means assignment.
//!
//! Uses per-series upper/lower bounds on DTW distances to skip unnecessary
//! distance computations during the assignment step.

use tracing::{debug, instrument};

use narcissus_dtw::{Dtw, TimeSeries};

use crate::inertia::Inertia;
use crate::label::ClusterLabel;

/// Elkan bounds state maintained across K-means iterations.
///
/// Tracks per-series upper bounds on distance to assigned centroid,
/// per-series-per-centroid lower bounds, and half inter-centroid distances.
/// On the first call to [`ElkanBounds::assign`], all distances are computed
/// exactly to initialize bounds; subsequent iterations exploit the triangle
/// inequality to skip redundant DTW evaluations.
#[derive(Debug)]
pub(crate) struct ElkanBounds {
    /// `upper[i]`: upper bound on dist(series[i], centroid[assigned[i]]).
    upper: Vec<f64>,
    /// `lower[i * k + c]`: lower bound on dist(series[i], centroid[c]).
    lower: Vec<f64>,
    /// Number of centroids.
    k: usize,
    /// Number of series.
    n: usize,
    /// Whether bounds have been initialized with exact distances.
    initialized: bool,
    /// Number of distance computations skipped due to bounds.
    skipped: usize,
}

impl ElkanBounds {
    /// Create fresh Elkan bounds for `n` series and `k` centroids.
    ///
    /// All upper bounds start at infinity and all lower bounds start at zero,
    /// ensuring no pruning occurs until exact distances are computed.
    pub(crate) fn new(n: usize, k: usize) -> Self {
        Self {
            upper: vec![f64::INFINITY; n],
            lower: vec![0.0; n * k],
            k,
            n,
            initialized: false,
            skipped: 0,
        }
    }

    /// Get the lower bound for series `i` vs centroid `c`.
    fn lower(&self, i: usize, c: usize) -> f64 {
        self.lower[i * self.k + c]
    }

    /// Set the lower bound for series `i` vs centroid `c`.
    fn set_lower(&mut self, i: usize, c: usize, val: f64) {
        self.lower[i * self.k + c] = val;
    }

    /// Perform Elkan-accelerated assignment.
    ///
    /// On the first invocation, computes all pairwise series-centroid distances
    /// exactly to initialize bounds. On subsequent calls, uses the triangle
    /// inequality to skip distance computations where bounds prove the current
    /// assignment cannot improve.
    ///
    /// Returns `(assignments, inertia, skip_count)`.
    #[instrument(skip(self, series, centroids, dtw, assignments),
                 fields(n = series.len(), k = centroids.len()))]
    pub(crate) fn assign(
        &mut self,
        series: &[TimeSeries],
        centroids: &[TimeSeries],
        dtw: &Dtw,
        assignments: &[usize],
    ) -> (Vec<ClusterLabel>, Inertia, usize) {
        self.skipped = 0;

        if !self.initialized {
            let result = self.assign_full(series, centroids, dtw);
            self.initialized = true;
            return result;
        }

        self.assign_pruned(series, centroids, dtw, assignments)
    }

    /// Compute all distances exactly (first iteration).
    fn assign_full(
        &mut self,
        series: &[TimeSeries],
        centroids: &[TimeSeries],
        dtw: &Dtw,
    ) -> (Vec<ClusterLabel>, Inertia, usize) {
        let mut new_assignments = vec![0usize; self.n];

        for (i, s) in series.iter().enumerate() {
            let view = s.as_view();
            let mut best_c = 0usize;
            let mut best_dist = f64::INFINITY;

            for (c, centroid) in centroids.iter().enumerate() {
                let d = dtw.distance(view, centroid.as_view()).value();
                self.set_lower(i, c, d);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }

            new_assignments[i] = best_c;
            self.upper[i] = best_dist;
        }

        let inertia_value: f64 = self.upper.iter().map(|d| d.powi(2)).sum();
        let labels = new_assignments
            .iter()
            .map(|&c| ClusterLabel::new(c))
            .collect();

        debug!(inertia = inertia_value, skipped = 0, "elkan full assignment complete");
        (labels, Inertia::new(inertia_value), 0)
    }

    /// Pruned assignment using triangle inequality bounds.
    fn assign_pruned(
        &mut self,
        series: &[TimeSeries],
        centroids: &[TimeSeries],
        dtw: &Dtw,
        assignments: &[usize],
    ) -> (Vec<ClusterLabel>, Inertia, usize) {
        // Compute inter-centroid distances and half-minimum per centroid.
        let inter = self.compute_inter_centroid(centroids, dtw);
        let half_min = self.compute_half_min_inter(&inter);

        let mut new_assignments = Vec::with_capacity(self.n);

        for (i, s) in series.iter().enumerate() {
            let mut a = assignments[i];

            // Rule 2: if upper bound <= half of min inter-centroid distance
            // for the assigned centroid, no other centroid can be closer.
            if self.upper[i] <= half_min[a] {
                new_assignments.push(a);
                self.skipped += 1;
                continue;
            }

            let view = s.as_view();
            let mut upper_tight = false;

            for c in 0..self.k {
                if c == a {
                    continue;
                }

                // Rule 1: if upper bound <= lower bound for this centroid, skip.
                if self.upper[i] <= self.lower(i, c) {
                    continue;
                }

                // Rule 3: if upper bound <= half inter-centroid distance
                // between assigned and candidate, skip.
                if self.upper[i] <= inter[a * self.k + c] / 2.0 {
                    continue;
                }

                // Tighten upper bound once if not already done this iteration.
                if !upper_tight {
                    let d_assigned = dtw.distance(view, centroids[a].as_view()).value();
                    self.upper[i] = d_assigned;
                    self.set_lower(i, a, d_assigned);
                    upper_tight = true;

                    // Re-check rules with tightened bound.
                    if self.upper[i] <= self.lower(i, c) {
                        continue;
                    }
                    if self.upper[i] <= inter[a * self.k + c] / 2.0 {
                        continue;
                    }
                }

                // Compute exact distance to candidate centroid.
                let d = dtw.distance(view, centroids[c].as_view()).value();
                self.set_lower(i, c, d);

                if d < self.upper[i] {
                    a = c;
                    self.upper[i] = d;
                }
            }

            new_assignments.push(a);
        }

        // Compute exact inertia: for series that were skipped entirely (Rule 2),
        // upper[i] may be inflated. Recompute exact distances for those.
        let inertia_value: f64 = series
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let c = new_assignments[i];
                let d = dtw.distance(s.as_view(), centroids[c].as_view()).value();
                // Also tighten the upper bound to the exact distance.
                self.upper[i] = d;
                self.set_lower(i, c, d);
                d.powi(2)
            })
            .sum();

        let labels = new_assignments
            .iter()
            .map(|&c| ClusterLabel::new(c))
            .collect();

        debug!(
            inertia = inertia_value,
            skipped = self.skipped,
            "elkan pruned assignment complete"
        );
        (labels, Inertia::new(inertia_value), self.skipped)
    }

    /// Compute the full inter-centroid distance matrix (k x k, stored flat).
    fn compute_inter_centroid(&self, centroids: &[TimeSeries], dtw: &Dtw) -> Vec<f64> {
        let k = centroids.len();
        let mut inter = vec![0.0; k * k];

        for c1 in 0..k {
            for c2 in (c1 + 1)..k {
                let d = dtw
                    .distance(centroids[c1].as_view(), centroids[c2].as_view())
                    .value();
                inter[c1 * k + c2] = d;
                inter[c2 * k + c1] = d;
            }
        }

        inter
    }

    /// Compute half of the minimum inter-centroid distance for each centroid.
    fn compute_half_min_inter(&self, inter: &[f64]) -> Vec<f64> {
        (0..self.k)
            .map(|c| {
                let min_dist = (0..self.k)
                    .filter(|&c2| c2 != c)
                    .map(|c2| inter[c * self.k + c2])
                    .fold(f64::INFINITY, f64::min);
                min_dist / 2.0
            })
            .collect()
    }

    /// Update bounds after centroid movement.
    ///
    /// Inflates upper bounds by the shift of the assigned centroid and deflates
    /// lower bounds by the shift of each candidate centroid (floored at zero).
    pub(crate) fn update_bounds(
        &mut self,
        centroid_shifts: &[f64],
        new_assignments: &[usize],
    ) {
        for i in 0..self.n {
            self.upper[i] += centroid_shifts[new_assignments[i]];
            for (c, &shift) in centroid_shifts.iter().enumerate() {
                let new_lower = (self.lower(i, c) - shift).max(0.0);
                self.set_lower(i, c, new_lower);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use narcissus_dtw::{BandConstraint, Dtw, TimeSeries};

    use super::ElkanBounds;
    use crate::config::KMeansConfig;
    use crate::kmeans::multi_restart;

    /// Nine series in three tight clusters near 0, 5, and 10.
    fn archetype_a() -> Vec<TimeSeries> {
        vec![
            TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![0.1, 0.0, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![0.0, 0.1, 0.0, 0.0]).unwrap(),
            TimeSeries::new(vec![5.0, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![5.1, 5.0, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![5.0, 5.1, 5.0, 5.0]).unwrap(),
            TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0]).unwrap(),
            TimeSeries::new(vec![10.1, 10.0, 10.0, 10.0]).unwrap(),
            TimeSeries::new(vec![10.0, 10.1, 10.0, 10.0]).unwrap(),
        ]
    }

    /// Normalize assignments so that the first seen label becomes 0, the next
    /// unseen label becomes 1, etc. This makes assignments comparable across
    /// runs that may permute cluster labels.
    fn normalize_labels(labels: &[usize]) -> Vec<usize> {
        let mut mapping = std::collections::HashMap::new();
        let mut next = 0usize;
        labels
            .iter()
            .map(|&l| {
                *mapping.entry(l).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect()
    }

    #[test]
    fn elkan_matches_standard_assign() {
        let series = archetype_a();

        let cfg_standard = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(42);
        let result_standard = multi_restart(&series, &cfg_standard, None).unwrap();

        let cfg_elkan = KMeansConfig::new(3, BandConstraint::Unconstrained)
            .unwrap()
            .with_n_init(5)
            .with_seed(42)
            .with_use_elkan(true);
        let result_elkan = multi_restart(&series, &cfg_elkan, None).unwrap();

        let labels_standard: Vec<usize> =
            result_standard.assignments.iter().map(|l| l.index()).collect();
        let labels_elkan: Vec<usize> =
            result_elkan.assignments.iter().map(|l| l.index()).collect();

        // Compare normalized labels (order-independent).
        let norm_standard = normalize_labels(&labels_standard);
        let norm_elkan = normalize_labels(&labels_elkan);

        assert_eq!(
            norm_standard, norm_elkan,
            "Elkan must produce the same cluster structure as standard K-means"
        );
        assert!(
            (result_standard.inertia.value() - result_elkan.inertia.value()).abs() < 1e-6,
            "inertia mismatch: standard={} elkan={}",
            result_standard.inertia.value(),
            result_elkan.inertia.value()
        );
    }

    #[test]
    fn elkan_skips_some_distances() {
        let series = archetype_a();
        let dtw = Dtw::unconstrained();
        let k = 3;
        let n = series.len();

        // Use centroids from well-separated clusters.
        let centroids = vec![
            series[0].clone(),
            series[3].clone(),
            series[6].clone(),
        ];

        let mut bounds = ElkanBounds::new(n, k);
        let initial_assignments = vec![0usize; n];

        // First call initializes bounds (no skips expected).
        let (labels1, _, skipped1) = bounds.assign(&series, &centroids, &dtw, &initial_assignments);
        assert_eq!(skipped1, 0, "first iteration should not skip any distances");

        // Extract assignments as usize for next call.
        let assignments1: Vec<usize> = labels1.iter().map(|l| l.index()).collect();

        // Simulate zero centroid movement (centroids unchanged).
        let zero_shifts = vec![0.0; k];
        bounds.update_bounds(&zero_shifts, &assignments1);

        // Second call should be able to skip some distances since bounds are tight
        // and centroids did not move.
        let (_, _, skipped2) = bounds.assign(&series, &centroids, &dtw, &assignments1);
        assert!(
            skipped2 > 0,
            "second iteration with unchanged centroids should skip some distances, got 0 skips"
        );
    }
}
