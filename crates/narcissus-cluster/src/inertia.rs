use std::cmp::Ordering;
use std::fmt;

/// Sum of squared DTW distances from each series to its assigned centroid.
///
/// Lower inertia indicates tighter clusters.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Inertia(f64);

impl Inertia {
    /// Create a new inertia value.
    pub(crate) fn new(value: f64) -> Self {
        Self(value)
    }

    /// Return the raw inertia value.
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

impl fmt::Display for Inertia {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::Inertia;

    #[test]
    fn roundtrip() {
        let inertia = Inertia::new(3.14);
        assert!((inertia.value() - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn display_format() {
        let inertia = Inertia::new(1.5);
        assert_eq!(format!("{inertia}"), "1.500000");
    }

    #[test]
    fn total_cmp_ordering() {
        let a = Inertia::new(1.0);
        let b = Inertia::new(2.0);
        assert_eq!(a.total_cmp(&b), Ordering::Less);
        assert_eq!(b.total_cmp(&a), Ordering::Greater);
        assert_eq!(a.total_cmp(&a), Ordering::Equal);
    }

    #[test]
    fn total_cmp_zero() {
        let zero = Inertia::new(0.0);
        let positive = Inertia::new(0.001);
        assert_eq!(zero.total_cmp(&positive), Ordering::Less);
    }
}
