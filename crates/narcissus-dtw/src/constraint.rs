//! Band constraint types for DTW computation.

use std::ops::Range;

/// Constraint on the DTW warping window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BandConstraint {
    /// No constraint — full cost matrix is computed.
    #[default]
    Unconstrained,

    /// Sakoe-Chiba band: cell (i,j) is valid only if |i - j| <= radius.
    SakoeChibaRadius(usize),
}

impl BandConstraint {
    /// Return the valid column range for a given row in the cost matrix.
    ///
    /// For unconstrained DTW, returns `0..n_cols`.
    /// For Sakoe-Chiba, returns the intersection of `[row - r, row + r]` with `[0, n_cols)`.
    #[must_use]
    pub fn column_range(&self, row: usize, n_cols: usize) -> Range<usize> {
        match self {
            Self::Unconstrained => 0..n_cols,
            Self::SakoeChibaRadius(r) => {
                let start = row.saturating_sub(*r);
                let end = (row + r + 1).min(n_cols);
                start..end
            }
        }
    }

    /// Return the maximum band width for a given matrix size.
    ///
    /// For unconstrained DTW, returns `m` (full width).
    /// For Sakoe-Chiba with radius `r`, returns `min(2*r + 1, m)`.
    #[must_use]
    pub fn band_width(&self, _n: usize, m: usize) -> usize {
        match self {
            Self::Unconstrained => m,
            Self::SakoeChibaRadius(r) => (2 * r + 1).min(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconstrained_full_range() {
        let c = BandConstraint::Unconstrained;
        assert_eq!(c.column_range(0, 10), 0..10);
        assert_eq!(c.column_range(5, 10), 0..10);
    }

    #[test]
    fn sakoe_chiba_middle_row() {
        let c = BandConstraint::SakoeChibaRadius(2);
        assert_eq!(c.column_range(5, 10), 3..8);
    }

    #[test]
    fn sakoe_chiba_first_row() {
        let c = BandConstraint::SakoeChibaRadius(2);
        assert_eq!(c.column_range(0, 10), 0..3);
    }

    #[test]
    fn sakoe_chiba_last_row() {
        let c = BandConstraint::SakoeChibaRadius(2);
        assert_eq!(c.column_range(9, 10), 7..10);
    }

    #[test]
    fn sakoe_chiba_radius_exceeds_size() {
        let c = BandConstraint::SakoeChibaRadius(20);
        assert_eq!(c.column_range(3, 5), 0..5);
    }

    #[test]
    fn band_width_unconstrained() {
        assert_eq!(BandConstraint::Unconstrained.band_width(10, 10), 10);
    }

    #[test]
    fn band_width_sakoe_chiba() {
        assert_eq!(BandConstraint::SakoeChibaRadius(2).band_width(52, 52), 5);
    }

    #[test]
    fn default_is_unconstrained() {
        assert_eq!(BandConstraint::default(), BandConstraint::Unconstrained);
    }
}
