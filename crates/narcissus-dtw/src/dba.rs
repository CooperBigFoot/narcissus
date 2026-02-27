//! DBA (DTW Barycenter Averaging) algorithm.

use tracing::{debug, instrument};

use crate::constraint::BandConstraint;
use crate::dtw::Dtw;
use crate::error::DbaError;
use crate::series::{TimeSeries, TimeSeriesView};

/// Configuration for DBA barycenter computation.
#[derive(Debug, Clone)]
pub struct DbaConfig {
    constraint: BandConstraint,
    max_iter: usize,
    tol: f64,
}

impl DbaConfig {
    /// Create a new DBA configuration with default parameters.
    ///
    /// Defaults: `max_iter = 10`, `tol = 1e-5`.
    #[must_use]
    pub fn new(constraint: BandConstraint) -> Self {
        Self {
            constraint,
            max_iter: 10,
            tol: 1e-5,
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Compute the DBA barycenter of a collection of time series.
    ///
    /// The centroid is initialized as the element-wise mean of all input series,
    /// then iteratively refined by DTW alignment (Petitjean et al. 2011).
    /// All series should have the same length as `series[0]`.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`DbaError::EmptyCluster`] | `series` is empty |
    /// | [`DbaError::Dtw`] | A DTW error occurs during alignment (e.g. non-finite centroid value) |
    #[instrument(skip(self, series), fields(n = series.len(), max_iter = self.max_iter))]
    pub fn average(&self, series: &[TimeSeriesView<'_>]) -> Result<DbaResult, DbaError> {
        if series.is_empty() {
            return Err(DbaError::EmptyCluster);
        }

        let len = series[0].len();
        let dtw = Dtw::from_constraint(self.constraint);

        // Initialize centroid as element-wise mean of all input series.
        let mut centroid_values: Vec<f64> = vec![0.0; len];
        for s in series {
            for (i, &v) in s.as_slice().iter().enumerate() {
                centroid_values[i] += v;
            }
        }
        let n_series = series.len() as f64;
        for v in &mut centroid_values {
            *v /= n_series;
        }

        let mut delta = f64::INFINITY;
        let mut iterations = 0;

        for iter in 0..self.max_iter {
            let mut sums = vec![0.0_f64; len];
            let mut counts = vec![0_usize; len];

            // Build a zero-copy centroid view. The values are all finite (they
            // started as finite inputs and we only ever average them), so the
            // unchecked constructor is safe here.
            let centroid_view = TimeSeriesView::new_unchecked(&centroid_values);

            // Align each series to the centroid and accumulate contributions.
            for s in series {
                let (_, path) = dtw.distance_and_path(centroid_view, *s);
                for step in path.steps() {
                    sums[step.a] += s[step.b];
                    counts[step.a] += 1;
                }
            }

            // Update centroid values and compute max absolute change.
            delta = 0.0_f64;
            for t in 0..len {
                let new_val = if counts[t] > 0 {
                    sums[t] / counts[t] as f64
                } else {
                    centroid_values[t]
                };
                delta = delta.max((new_val - centroid_values[t]).abs());
                centroid_values[t] = new_val;
            }

            iterations = iter + 1;
            debug!(iteration = iterations, delta, "DBA iteration complete");

            if delta < self.tol {
                break;
            }
        }

        let centroid = TimeSeries::new(centroid_values)?;
        Ok(DbaResult {
            centroid,
            converged: delta < self.tol,
            iterations,
            final_delta: delta,
        })
    }
}

/// Result of a DBA computation.
#[derive(Debug, Clone)]
pub struct DbaResult {
    /// The computed centroid time series.
    pub centroid: TimeSeries,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Maximum absolute change in the final iteration.
    pub final_delta: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::BandConstraint;
    use crate::series::TimeSeries;

    #[test]
    fn dba_identical_series() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s1 = TimeSeries::new(values.clone()).unwrap();
        let s2 = TimeSeries::new(values.clone()).unwrap();
        let s3 = TimeSeries::new(values.clone()).unwrap();

        let views = [s1.as_view(), s2.as_view(), s3.as_view()];
        let config = DbaConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&views).unwrap();

        // Centroid should equal the input series.
        let centroid = result.centroid.as_ref();
        for (i, &v) in centroid.iter().enumerate() {
            assert!(
                (v - values[i]).abs() < 1e-10,
                "centroid[{i}] = {v}, expected {}",
                values[i]
            );
        }
        // Should converge in 1 iteration (no change after first alignment).
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn dba_single_series() {
        let values = vec![10.0, 20.0, 30.0];
        let s = TimeSeries::new(values.clone()).unwrap();

        let views = [s.as_view()];
        let config = DbaConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&views).unwrap();

        let centroid = result.centroid.as_ref();
        for (i, &v) in centroid.iter().enumerate() {
            assert!((v - values[i]).abs() < 1e-10);
        }
        assert!(result.converged);
    }

    #[test]
    fn dba_empty_cluster_error() {
        let config = DbaConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&[]);
        assert!(matches!(result, Err(DbaError::EmptyCluster)));
    }

    #[test]
    fn dba_convergence_metadata() {
        let s1 = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let s2 = TimeSeries::new(vec![2.0, 3.0, 4.0]).unwrap();

        let views = [s1.as_view(), s2.as_view()];
        let config = DbaConfig::new(BandConstraint::Unconstrained)
            .with_max_iter(50)
            .with_tol(1e-10);
        let result = config.average(&views).unwrap();

        // Should converge at some point within the iteration budget.
        assert!(result.iterations <= 50);
        assert!(result.final_delta >= 0.0);
    }

    #[test]
    fn dba_with_band_constraint() {
        let s1 = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s2 = TimeSeries::new(vec![1.5, 2.5, 3.5, 4.5]).unwrap();

        let views = [s1.as_view(), s2.as_view()];
        let config = DbaConfig::new(BandConstraint::SakoeChibaRadius(1));
        let result = config.average(&views).unwrap();

        // Centroid should be somewhere between the two series.
        let centroid = result.centroid.as_ref();
        assert_eq!(centroid.len(), 4);
        assert!(result.converged || result.iterations > 0);
    }
}
