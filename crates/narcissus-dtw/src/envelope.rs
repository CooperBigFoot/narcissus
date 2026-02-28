//! Sakoe-Chiba envelope computation and DTW lower bounds.

use std::collections::VecDeque;

use crate::constraint::BandConstraint;
use crate::series::TimeSeriesView;

/// Precomputed upper and lower Sakoe-Chiba envelope for a time series.
///
/// For each time step `i`, `upper[i]` is the maximum of `series[j]` where `|i - j| <= radius`,
/// and `lower[i]` is the minimum. For unconstrained DTW, the envelopes are the global
/// max/min repeated.
#[derive(Debug, Clone)]
pub struct SeriesEnvelope {
    upper: Vec<f64>,
    lower: Vec<f64>,
}

impl SeriesEnvelope {
    /// Compute upper and lower envelopes for a series under the given constraint.
    ///
    /// Uses an O(n) sliding-window min/max algorithm with [`VecDeque`].
    /// For [`BandConstraint::Unconstrained`], upper and lower are the global max/min.
    #[must_use]
    pub fn compute(series: TimeSeriesView<'_>, constraint: BandConstraint) -> Self {
        let data = series.as_slice();
        let n = data.len();

        let radius = match constraint {
            BandConstraint::Unconstrained => n, // full window = global extremes
            BandConstraint::SakoeChibaRadius(r) => r,
        };

        let mut upper = vec![0.0_f64; n];
        let mut lower = vec![0.0_f64; n];

        // Sliding centered-window max and min using monotonic deques.
        //
        // For each query index i, the window is [max(0, i-radius), min(n-1, i+radius)].
        // We iterate i from 0..n, adding elements up to min(i+radius, n-1) before
        // querying, and evicting elements before max(0, i-radius) from the front.
        //
        // Max deque invariant: indices in deque are in increasing order, values are
        // in decreasing order (front = current window maximum).
        //
        // Min deque invariant: indices in deque are in increasing order, values are
        // in increasing order (front = current window minimum).
        let mut max_deque: VecDeque<usize> = VecDeque::new();
        let mut min_deque: VecDeque<usize> = VecDeque::new();
        let mut next_to_add: usize = 0;

        for i in 0..n {
            let hi = (i + radius).min(n - 1);

            // Add new elements up to hi into both deques.
            while next_to_add <= hi {
                // Max deque: pop smaller-or-equal values from the back so the
                // back always holds an index with a value larger than what we add.
                while let Some(&back) = max_deque.back() {
                    if data[back] <= data[next_to_add] {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back(next_to_add);

                // Min deque: pop larger-or-equal values from the back.
                while let Some(&back) = min_deque.back() {
                    if data[back] >= data[next_to_add] {
                        min_deque.pop_back();
                    } else {
                        break;
                    }
                }
                min_deque.push_back(next_to_add);

                next_to_add += 1;
            }

            let lo = i.saturating_sub(radius);

            // Evict elements that have fallen outside the left boundary of the window.
            while let Some(&front) = max_deque.front() {
                if front < lo {
                    max_deque.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&front) = min_deque.front() {
                if front < lo {
                    min_deque.pop_front();
                } else {
                    break;
                }
            }

            // SAFETY: at least element `i` itself is always in the window, so
            // the deques are never empty at this point.
            upper[i] = data[*max_deque.front().expect("max_deque must be non-empty")];
            lower[i] = data[*min_deque.front().expect("min_deque must be non-empty")];
        }

        Self { upper, lower }
    }

    /// Return the upper envelope values.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Return the lower envelope values.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Return the length of the envelope (same as the original series).
    #[must_use]
    pub fn len(&self) -> usize {
        self.upper.len()
    }

    /// Return true if the envelope is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.upper.is_empty()
    }
}

/// Compute the LB_Keogh lower bound on DTW distance.
///
/// For each time step `i` in `query`, if `query[i]` falls outside the envelope of
/// the candidate, the squared difference to the nearest envelope boundary is
/// accumulated. Returns the square root of the total.
///
/// This is a valid lower bound: `lb_keogh(q, env_c) <= dtw(q, c)`.
#[must_use]
pub fn lb_keogh(query: &[f64], envelope: &SeriesEnvelope) -> f64 {
    let n = query.len().min(envelope.len());
    let sum_sq = query[..n]
        .iter()
        .zip(envelope.upper[..n].iter().zip(envelope.lower[..n].iter()))
        .map(|(&q, (&u, &l))| {
            if q > u {
                let diff = q - u;
                diff * diff
            } else if q < l {
                let diff = l - q;
                diff * diff
            } else {
                0.0
            }
        })
        .sum::<f64>();
    sum_sq.sqrt()
}

/// Compute the LB_Improved lower bound (Lemire 2009).
///
/// Uses envelopes from both query and candidate to produce a tighter lower bound
/// than [`lb_keogh`] alone. Always `>= lb_keogh(query, candidate_envelope)`.
///
/// Algorithm:
/// 1. For positions where the query is *outside* the candidate envelope, accumulate
///    the standard LB_Keogh squared contribution for the query, then additionally
///    check whether the boundary point that was touched falls outside the *query*
///    envelope and accumulate that extra squared gap.
/// 2. For positions where the query is *inside* the candidate envelope, check
///    whether the candidate value itself falls outside the query envelope and
///    accumulate that squared gap.
/// 3. Return `sqrt(total_sum_sq)`.
#[must_use]
pub fn lb_improved(
    query: &[f64],
    query_envelope: &SeriesEnvelope,
    candidate: &[f64],
    candidate_envelope: &SeriesEnvelope,
) -> f64 {
    let n = query
        .len()
        .min(candidate.len())
        .min(query_envelope.len())
        .min(candidate_envelope.len());

    let sum_sq: f64 = (0..n)
        .map(|i| {
            let q = query[i];
            let c = candidate[i];
            let cu = candidate_envelope.upper[i];
            let cl = candidate_envelope.lower[i];
            let qu = query_envelope.upper[i];
            let ql = query_envelope.lower[i];

            if q > cu {
                // Query above candidate envelope — LB_Keogh contribution from query side.
                let keogh_diff = q - cu;
                let keogh_sq = keogh_diff * keogh_diff;

                // Improved contribution: project candidate upper boundary onto query envelope.
                let proj = cu;
                let improved_sq = if proj > qu {
                    let d = proj - qu;
                    d * d
                } else if proj < ql {
                    let d = ql - proj;
                    d * d
                } else {
                    0.0
                };

                keogh_sq + improved_sq
            } else if q < cl {
                // Query below candidate envelope — LB_Keogh contribution from query side.
                let keogh_diff = cl - q;
                let keogh_sq = keogh_diff * keogh_diff;

                // Improved contribution: project candidate lower boundary onto query envelope.
                let proj = cl;
                let improved_sq = if proj > qu {
                    let d = proj - qu;
                    d * d
                } else if proj < ql {
                    let d = ql - proj;
                    d * d
                } else {
                    0.0
                };

                keogh_sq + improved_sq
            } else {
                // Query inside candidate envelope — only improved contribution from candidate side.
                if c > qu {
                    let d = c - qu;
                    d * d
                } else if c < ql {
                    let d = ql - c;
                    d * d
                } else {
                    0.0
                }
            }
        })
        .sum();

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dtw::Dtw;
    use crate::series::TimeSeries;

    /// Five (query, candidate) pairs used across multiple tests.
    fn test_pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]),
            (vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 2.0, 3.0, 4.0]),
            (vec![1.0, 3.0, 2.0, 5.0, 4.0], vec![2.0, 1.0, 4.0, 3.0, 6.0]),
            (vec![10.0, -10.0, 10.0, -10.0], vec![-10.0, 10.0, -10.0, 10.0]),
            (vec![1.0, 1.5, 2.0, 2.5, 3.0], vec![1.0, 1.5, 2.0, 2.5, 3.0]),
        ]
    }

    #[test]
    fn lb_keogh_leq_dtw() {
        let dtw = Dtw::with_sakoe_chiba(2);
        let eps = 1e-9;

        for (q_vec, c_vec) in test_pairs() {
            let q_ts = TimeSeries::new(q_vec.clone()).unwrap();
            let c_ts = TimeSeries::new(c_vec.clone()).unwrap();

            let envelope = SeriesEnvelope::compute(c_ts.as_view(), BandConstraint::SakoeChibaRadius(2));
            let lb = lb_keogh(&q_vec, &envelope);
            let dtw_dist = dtw.distance(q_ts.as_view(), c_ts.as_view()).value();

            assert!(
                lb <= dtw_dist + eps,
                "lb_keogh ({lb}) > dtw ({dtw_dist}) for query {q_vec:?} vs candidate {c_vec:?}"
            );
        }
    }

    #[test]
    fn lb_improved_geq_lb_keogh() {
        let eps = 1e-9;

        for (q_vec, c_vec) in test_pairs() {
            let q_ts = TimeSeries::new(q_vec.clone()).unwrap();
            let c_ts = TimeSeries::new(c_vec.clone()).unwrap();

            let q_env = SeriesEnvelope::compute(q_ts.as_view(), BandConstraint::SakoeChibaRadius(2));
            let c_env = SeriesEnvelope::compute(c_ts.as_view(), BandConstraint::SakoeChibaRadius(2));

            let lb_k = lb_keogh(&q_vec, &c_env);
            let lb_i = lb_improved(&q_vec, &q_env, &c_vec, &c_env);

            assert!(
                lb_i >= lb_k - eps,
                "lb_improved ({lb_i}) < lb_keogh ({lb_k}) for query {q_vec:?} vs candidate {c_vec:?}"
            );
        }
    }

    #[test]
    fn lb_keogh_identical_series_zero() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(series.clone()).unwrap();
        let envelope = SeriesEnvelope::compute(ts.as_view(), BandConstraint::SakoeChibaRadius(2));
        let lb = lb_keogh(&series, &envelope);
        assert!(
            lb.abs() < 1e-12,
            "expected lb_keogh == 0 for identical series, got {lb}"
        );
    }

    #[test]
    fn envelope_upper_geq_lower() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        let ts = TimeSeries::new(data).unwrap();
        let envelope = SeriesEnvelope::compute(ts.as_view(), BandConstraint::SakoeChibaRadius(2));

        for i in 0..envelope.len() {
            assert!(
                envelope.upper[i] >= envelope.lower[i],
                "upper[{i}] ({}) < lower[{i}] ({})",
                envelope.upper[i],
                envelope.lower[i]
            );
        }
    }

    #[test]
    fn envelope_unconstrained_is_global_extremes() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let ts = TimeSeries::new(data.clone()).unwrap();
        let envelope = SeriesEnvelope::compute(ts.as_view(), BandConstraint::Unconstrained);

        let global_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let global_min = data.iter().cloned().fold(f64::INFINITY, f64::min);

        for i in 0..envelope.len() {
            assert!(
                (envelope.upper[i] - global_max).abs() < 1e-12,
                "upper[{i}] ({}) != global_max ({})",
                envelope.upper[i],
                global_max
            );
            assert!(
                (envelope.lower[i] - global_min).abs() < 1e-12,
                "lower[{i}] ({}) != global_min ({})",
                envelope.lower[i],
                global_min
            );
        }
    }

    #[test]
    fn envelope_radius_zero_equals_series() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let ts = TimeSeries::new(data.clone()).unwrap();
        let envelope = SeriesEnvelope::compute(ts.as_view(), BandConstraint::SakoeChibaRadius(0));

        for i in 0..envelope.len() {
            assert!(
                (envelope.upper[i] - data[i]).abs() < 1e-12,
                "upper[{i}] ({}) != data[{i}] ({})",
                envelope.upper[i],
                data[i]
            );
            assert!(
                (envelope.lower[i] - data[i]).abs() < 1e-12,
                "lower[{i}] ({}) != data[{i}] ({})",
                envelope.lower[i],
                data[i]
            );
        }
    }
}
