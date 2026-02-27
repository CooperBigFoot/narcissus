use std::fmt;

/// A cluster assignment label. Wraps a zero-based cluster index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterLabel(usize);

impl ClusterLabel {
    /// Create a new cluster label from a zero-based index.
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based cluster index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for ClusterLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::ClusterLabel;

    #[test]
    fn roundtrip() {
        let label = ClusterLabel::new(7);
        assert_eq!(label.index(), 7);
    }

    #[test]
    fn display() {
        let label = ClusterLabel::new(3);
        assert_eq!(format!("{label}"), "3");
    }

    #[test]
    fn ordering() {
        let a = ClusterLabel::new(1);
        let b = ClusterLabel::new(5);
        assert!(a < b);
    }

    #[test]
    fn equality() {
        let a = ClusterLabel::new(2);
        let b = ClusterLabel::new(2);
        assert_eq!(a, b);
    }
}
