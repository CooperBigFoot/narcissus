//! Stochastic Subgradient (SSG) centroid averaging.
//!
//! An alternative to DBA that updates the centroid one series at a time
//! with a decaying learning rate, trading per-iteration cost for
//! potentially faster convergence on large datasets.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tracing::{debug, instrument};

use crate::constraint::BandConstraint;
use crate::dtw::Dtw;
use crate::error::DbaError;
use crate::series::{TimeSeries, TimeSeriesView};

/// Configuration for SSG centroid averaging.
#[derive(Debug, Clone)]
pub struct SsgConfig {
    constraint: BandConstraint,
    max_epochs: usize,
    lr_init: f64,
    decay: f64,
    tol: f64,
    seed: u64,
}

impl SsgConfig {
    /// Create a new SSG configuration with default parameters.
    ///
    /// Defaults: `max_epochs = 20`, `lr_init = 0.1`, `decay = 0.01`, `tol = 1e-5`, `seed = 42`.
    #[must_use]
    pub fn new(constraint: BandConstraint) -> Self {
        Self {
            constraint,
            max_epochs: 20,
            lr_init: 0.1,
            decay: 0.01,
            tol: 1e-5,
            seed: 42,
        }
    }

    /// Set the maximum number of epochs.
    #[must_use]
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the initial learning rate.
    #[must_use]
    pub fn with_lr_init(mut self, lr_init: f64) -> Self {
        self.lr_init = lr_init;
        self
    }

    /// Set the learning rate decay factor.
    #[must_use]
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the random seed for shuffling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Compute the SSG centroid of a collection of time series.
    ///
    /// For each epoch: shuffle the series, then for each series, align it to
    /// the current centroid via DTW, update the centroid with a decaying
    /// learning rate: `centroid[a] += lr * (series[b] - centroid[a])`.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`DbaError::EmptyCluster`] | `series` is empty |
    /// | [`DbaError::Dtw`] | DTW error during alignment |
    #[instrument(skip(self, series), fields(n = series.len(), max_epochs = self.max_epochs))]
    pub fn average(&self, series: &[TimeSeriesView<'_>]) -> Result<SsgResult, DbaError> {
        if series.is_empty() {
            return Err(DbaError::EmptyCluster);
        }

        let len = series[0].len();
        let dtw = Dtw::from_constraint(self.constraint);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        // Initialize centroid as element-wise mean
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

        let mut global_step: usize = 0;
        let mut converged = false;
        let mut epochs_done = 0;

        for epoch in 0..self.max_epochs {
            let mut max_delta = 0.0_f64;

            // Shuffle series order for this epoch
            let mut order: Vec<usize> = (0..series.len()).collect();
            order.shuffle(&mut rng);

            for &idx in &order {
                global_step += 1;
                let lr = self.lr_init / (1.0 + self.decay * global_step as f64);

                let centroid_view = TimeSeriesView::new_unchecked(&centroid_values);
                let (_, path) = dtw.distance_and_path(centroid_view, series[idx]);

                // Update centroid based on alignment
                for step in path.steps() {
                    let old = centroid_values[step.a];
                    let target = series[idx][step.b];
                    let new_val = old + lr * (target - old);
                    let delta = (new_val - old).abs();
                    max_delta = max_delta.max(delta);
                    centroid_values[step.a] = new_val;
                }
            }

            epochs_done = epoch + 1;
            debug!(
                epoch = epochs_done,
                max_delta,
                lr_current = self.lr_init / (1.0 + self.decay * global_step as f64),
                "SSG epoch complete"
            );

            if max_delta < self.tol {
                converged = true;
                break;
            }
        }

        let centroid = TimeSeries::new(centroid_values)?;
        Ok(SsgResult {
            centroid,
            converged,
            epochs: epochs_done,
            global_steps: global_step,
        })
    }
}

/// Result of an SSG centroid computation.
#[derive(Debug, Clone)]
pub struct SsgResult {
    /// The computed centroid time series.
    pub centroid: TimeSeries,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// Number of epochs performed.
    pub epochs: usize,
    /// Total number of individual update steps.
    pub global_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::BandConstraint;
    use crate::error::DbaError;
    use crate::series::TimeSeries;

    #[test]
    fn identical_series() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s1 = TimeSeries::new(values.clone()).unwrap();
        let s2 = TimeSeries::new(values.clone()).unwrap();
        let s3 = TimeSeries::new(values.clone()).unwrap();

        let views = [s1.as_view(), s2.as_view(), s3.as_view()];
        let config = SsgConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&views).unwrap();

        let centroid = result.centroid.as_ref();
        for (i, &v) in centroid.iter().enumerate() {
            assert!(
                (v - values[i]).abs() < 1e-6,
                "centroid[{i}] = {v}, expected {}",
                values[i]
            );
        }
    }

    #[test]
    fn single_series() {
        let values = vec![10.0, 20.0, 30.0];
        let s = TimeSeries::new(values.clone()).unwrap();

        let views = [s.as_view()];
        let config = SsgConfig::new(BandConstraint::Unconstrained)
            .with_max_epochs(50)
            .with_tol(1e-10);
        let result = config.average(&views).unwrap();

        let centroid = result.centroid.as_ref();
        for (i, &v) in centroid.iter().enumerate() {
            assert!(
                (v - values[i]).abs() < 1e-6,
                "centroid[{i}] = {v}, expected {}",
                values[i]
            );
        }
    }

    #[test]
    fn empty_error() {
        let config = SsgConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&[]);
        assert!(matches!(result, Err(DbaError::EmptyCluster)));
    }

    #[test]
    fn deterministic() {
        let s1 = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s2 = TimeSeries::new(vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        let s3 = TimeSeries::new(vec![0.0, 1.0, 2.0, 3.0]).unwrap();

        let views = [s1.as_view(), s2.as_view(), s3.as_view()];

        let config = SsgConfig::new(BandConstraint::Unconstrained).with_seed(99);
        let result1 = config.average(&views).unwrap();

        let config2 = SsgConfig::new(BandConstraint::Unconstrained).with_seed(99);
        let result2 = config2.average(&views).unwrap();

        let c1 = result1.centroid.as_ref();
        let c2 = result2.centroid.as_ref();
        for (v1, v2) in c1.iter().zip(c2.iter()) {
            assert_eq!(v1, v2, "runs with same seed produced different results");
        }
    }

    #[test]
    fn centroid_between_inputs() {
        let s1 = TimeSeries::new(vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let s2 = TimeSeries::new(vec![10.0, 10.0, 10.0, 10.0]).unwrap();

        let views = [s1.as_view(), s2.as_view()];
        let config = SsgConfig::new(BandConstraint::Unconstrained);
        let result = config.average(&views).unwrap();

        let centroid = result.centroid.as_ref();
        for &v in centroid {
            assert!(
                v > 0.0 && v < 10.0,
                "centroid value {v} is not between 0.0 and 10.0"
            );
        }
    }
}
