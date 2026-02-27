//! Warping path types for DTW alignment.

/// A single step in a DTW warping path, mapping index `a` in the first series
/// to index `b` in the second series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpingStep {
    /// Index in the first time series.
    pub a: usize,
    /// Index in the second time series.
    pub b: usize,
}

/// An ordered sequence of warping steps from `(0, 0)` to `(n-1, m-1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct WarpingPath(Vec<WarpingStep>);

impl WarpingPath {
    /// Create a new warping path from a vector of steps.
    pub(crate) fn new(steps: Vec<WarpingStep>) -> Self {
        Self(steps)
    }

    /// Return the warping steps as a slice.
    #[must_use]
    pub fn steps(&self) -> &[WarpingStep] {
        &self.0
    }

    /// Return the number of steps in the path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the path contains no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'a> IntoIterator for &'a WarpingPath {
    type Item = &'a WarpingStep;
    type IntoIter = std::slice::Iter<'a, WarpingStep>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
