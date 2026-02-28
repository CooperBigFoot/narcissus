//! DTW distance newtype wrapper.

use std::cmp::Ordering;
use std::fmt;

/// A non-negative DTW distance value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DtwDistance(f64);

impl DtwDistance {
    /// Infinite DTW distance, used as a sentinel when early abandoning.
    pub const INFINITY: Self = Self(f64::INFINITY);

    /// Create a new DTW distance from a raw value.
    pub(crate) fn new(value: f64) -> Self {
        Self(value)
    }

    /// Return the raw distance value.
    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }

    /// Total ordering comparison using [`f64::total_cmp`].
    #[must_use]
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl fmt::Display for DtwDistance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_format() {
        let d = DtwDistance::new(1.234567);
        assert_eq!(format!("{d}"), "1.234567");
    }

    #[test]
    fn total_cmp_ordering() {
        let a = DtwDistance::new(1.0);
        let b = DtwDistance::new(2.0);
        assert_eq!(a.total_cmp(&b), Ordering::Less);
        assert_eq!(b.total_cmp(&a), Ordering::Greater);
        assert_eq!(a.total_cmp(&a), Ordering::Equal);
    }

    #[test]
    fn value_roundtrip() {
        let d = DtwDistance::new(42.0);
        assert_eq!(d.value(), 42.0);
    }
}
