//! Time series preprocessing: z-normalization and derivative transform.

use crate::error::{DerivativeError, PreprocessError};
use crate::series::TimeSeries;

/// Z-normalize a time series to zero mean and unit variance.
///
/// Uses population standard deviation (divides by n, not n-1).
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`PreprocessError::ConstantSeries`] | All values are identical (zero variance) |
#[must_use = "returns a new normalized series; the original is unchanged"]
pub fn z_normalize(series: &TimeSeries) -> Result<TimeSeries, PreprocessError> {
    let data = series.as_ref();
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return Err(PreprocessError::ConstantSeries {
            n: data.len(),
            value: data[0],
        });
    }

    let normalized: Vec<f64> = data.iter().map(|&x| (x - mean) / std).collect();
    // z-normalized values are always finite when input is finite and std > 0
    Ok(TimeSeries::new(normalized).expect("z-normalized values should be finite"))
}

/// Z-normalize a batch of time series.
///
/// Each series is independently normalized to zero mean and unit variance.
///
/// # Errors
///
/// Returns the first [`PreprocessError`] encountered.
#[must_use = "returns a new vector of normalized series"]
pub fn z_normalize_batch(series: &[TimeSeries]) -> Result<Vec<TimeSeries>, PreprocessError> {
    series.iter().map(z_normalize).collect()
}

/// Compute the Keogh-Pazzani first derivative of a time series.
///
/// For interior points (1..n-1): `d[i] = ((x[i] - x[i-1]) + (x[i+1] - x[i-1]) / 2) / 2`
/// Output length is `n - 2` (drops first and last points).
///
/// # Errors
///
/// | Variant | Condition |
/// |---|---|
/// | [`DerivativeError::TooShort`] | Series has fewer than 3 elements |
#[must_use = "returns a new derivative series; the original is unchanged"]
pub fn derivative(series: &TimeSeries) -> Result<TimeSeries, DerivativeError> {
    let data = series.as_ref();
    let n = data.len();

    if n < 3 {
        return Err(DerivativeError::TooShort { len: n });
    }

    let deriv: Vec<f64> = (1..n - 1)
        .map(|i| ((data[i] - data[i - 1]) + (data[i + 1] - data[i - 1]) / 2.0) / 2.0)
        .collect();

    // Derivative of finite values is always finite
    Ok(TimeSeries::new(deriv).expect("derivative values should be finite"))
}

#[cfg(test)]
mod tests {
    use crate::error::{DerivativeError, PreprocessError};
    use crate::series::TimeSeries;

    use super::*;

    fn make_series(values: Vec<f64>) -> TimeSeries {
        TimeSeries::new(values).unwrap()
    }

    #[test]
    fn z_normalize_zero_mean() {
        let ts = make_series(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = z_normalize(&ts).unwrap();
        let data = normalized.as_ref();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 1e-10, "mean was {mean}");
    }

    #[test]
    fn z_normalize_unit_variance() {
        let ts = make_series(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = z_normalize(&ts).unwrap();
        let data = normalized.as_ref();
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((variance - 1.0).abs() < 1e-10, "variance was {variance}");
    }

    #[test]
    fn z_normalize_constant_series_error() {
        let ts = make_series(vec![5.0, 5.0, 5.0]);
        let result = z_normalize(&ts);
        assert!(
            matches!(result, Err(PreprocessError::ConstantSeries { n: 3, value: 5.0 })),
            "expected ConstantSeries error, got {result:?}"
        );
    }

    #[test]
    fn z_normalize_batch_all_succeed() {
        let batch = vec![
            make_series(vec![1.0, 2.0, 3.0]),
            make_series(vec![10.0, 20.0, 30.0]),
            make_series(vec![-1.0, 0.0, 1.0]),
        ];
        let result = z_normalize_batch(&batch);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn z_normalize_batch_one_constant_fails() {
        let batch = vec![
            make_series(vec![1.0, 2.0, 3.0]),
            make_series(vec![7.0, 7.0, 7.0]),
            make_series(vec![4.0, 5.0, 6.0]),
        ];
        let result = z_normalize_batch(&batch);
        assert!(
            matches!(result, Err(PreprocessError::ConstantSeries { .. })),
            "expected ConstantSeries error, got {result:?}"
        );
    }

    #[test]
    fn derivative_length() {
        let ts = make_series(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let d = derivative(&ts).unwrap();
        assert_eq!(d.len(), 3, "expected length 3 for input of length 5");
    }

    #[test]
    fn derivative_too_short() {
        let ts = make_series(vec![1.0, 2.0]);
        let result = derivative(&ts);
        assert!(
            matches!(result, Err(DerivativeError::TooShort { len: 2 })),
            "expected TooShort error, got {result:?}"
        );
    }

    #[test]
    fn derivative_linear_series() {
        let ts = make_series(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let d = derivative(&ts).unwrap();
        let data = d.as_ref();
        for &v in data {
            assert!((v - 1.0).abs() < 1e-10, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn derivative_values_finite() {
        let ts = make_series(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]);
        let d = derivative(&ts).unwrap();
        for &v in d.as_ref() {
            assert!(v.is_finite(), "expected finite value, got {v}");
        }
    }
}
