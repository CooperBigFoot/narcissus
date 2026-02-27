//! Time series types with validation guarantees.

use std::ops::Index;

use crate::error::DtwError;

/// Owned, validated time series. Guaranteed non-empty with all finite values.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSeries(Vec<f64>);

impl TimeSeries {
    /// Create a new time series, validating that it is non-empty and all values are finite.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`DtwError::EmptySeries`] | `values` is empty |
    /// | [`DtwError::NonFiniteValue`] | Any value is NaN or infinite |
    pub fn new(values: Vec<f64>) -> Result<Self, DtwError> {
        if values.is_empty() {
            return Err(DtwError::EmptySeries);
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(DtwError::NonFiniteValue { index });
        }
        Ok(Self(values))
    }

    /// Borrow this series as a zero-copy view.
    #[must_use]
    pub fn as_view(&self) -> TimeSeriesView<'_> {
        TimeSeriesView::new_unchecked(&self.0)
    }

    /// Return the number of time steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the series has no time steps.
    ///
    /// A [`TimeSeries`] constructed via [`TimeSeries::new`] is always non-empty,
    /// so this always returns `false` for valid instances. Provided to satisfy
    /// the `len_without_is_empty` convention.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consume and return the inner vector.
    #[must_use]
    pub fn into_inner(self) -> Vec<f64> {
        self.0
    }
}

impl AsRef<[f64]> for TimeSeries {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

impl TryFrom<Vec<f64>> for TimeSeries {
    type Error = DtwError;

    fn try_from(values: Vec<f64>) -> Result<Self, Self::Error> {
        Self::new(values)
    }
}

/// Borrowed, validated view into a time series. Zero-copy reference.
#[derive(Debug, Clone, Copy)]
pub struct TimeSeriesView<'a>(&'a [f64]);

impl<'a> TimeSeriesView<'a> {
    /// Create a new view, validating that the slice is non-empty and all values are finite.
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`DtwError::EmptySeries`] | `slice` is empty |
    /// | [`DtwError::NonFiniteValue`] | Any value is NaN or infinite |
    pub fn new(slice: &'a [f64]) -> Result<Self, DtwError> {
        if slice.is_empty() {
            return Err(DtwError::EmptySeries);
        }
        if let Some(index) = slice.iter().position(|v| !v.is_finite()) {
            return Err(DtwError::NonFiniteValue { index });
        }
        Ok(Self(slice))
    }

    /// Create a view without validation. For internal use where data is already validated.
    pub(crate) fn new_unchecked(slice: &'a [f64]) -> Self {
        Self(slice)
    }

    /// Return the underlying slice.
    #[must_use]
    pub fn as_slice(&self) -> &'a [f64] {
        self.0
    }

    /// Return the number of time steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the view has no time steps.
    ///
    /// A [`TimeSeriesView`] constructed via [`TimeSeriesView::new`] is always non-empty,
    /// so this always returns `false` for valid instances. Provided to satisfy
    /// the `len_without_is_empty` convention.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Index<usize> for TimeSeriesView<'_> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl AsRef<[f64]> for TimeSeriesView<'_> {
    fn as_ref(&self) -> &[f64] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_vec() {
        let result = TimeSeries::new(vec![]);
        assert!(matches!(result, Err(DtwError::EmptySeries)));
    }

    #[test]
    fn rejects_nan() {
        let result = TimeSeries::new(vec![1.0, f64::NAN, 3.0]);
        assert!(matches!(result, Err(DtwError::NonFiniteValue { index: 1 })));
    }

    #[test]
    fn rejects_infinity() {
        let result = TimeSeries::new(vec![1.0, 2.0, f64::INFINITY]);
        assert!(matches!(result, Err(DtwError::NonFiniteValue { index: 2 })));
    }

    #[test]
    fn rejects_neg_infinity() {
        let result = TimeSeries::new(vec![f64::NEG_INFINITY, 2.0]);
        assert!(matches!(result, Err(DtwError::NonFiniteValue { index: 0 })));
    }

    #[test]
    fn accepts_valid_series() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.as_ref(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn view_rejects_empty() {
        let result = TimeSeriesView::new(&[]);
        assert!(matches!(result, Err(DtwError::EmptySeries)));
    }

    #[test]
    fn view_rejects_nan() {
        let data = [1.0, f64::NAN];
        let result = TimeSeriesView::new(&data);
        assert!(matches!(result, Err(DtwError::NonFiniteValue { index: 1 })));
    }

    #[test]
    fn view_indexing() {
        let data = [10.0, 20.0, 30.0];
        let view = TimeSeriesView::new(&data).unwrap();
        assert_eq!(view[0], 10.0);
        assert_eq!(view[2], 30.0);
    }

    #[test]
    fn try_from_vec() {
        let ts: Result<TimeSeries, _> = vec![1.0, 2.0].try_into();
        assert!(ts.is_ok());
    }

    #[test]
    fn as_view_roundtrip() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0]).unwrap();
        let view = ts.as_view();
        assert_eq!(view.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
