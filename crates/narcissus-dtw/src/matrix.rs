//! Lower-triangular distance matrix for pairwise DTW distances.

use std::ops::Index;

use crate::distance::DtwDistance;

/// Symmetric distance matrix stored as a lower-triangular flat vector.
///
/// For `n` series, stores `n*(n-1)/2` distances. Access is symmetric:
/// `get(i, j) == get(j, i)`. Diagonal is always zero.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    n: usize,
    data: Vec<DtwDistance>,
}

impl DistanceMatrix {
    /// Create a new distance matrix from pre-computed lower-triangular data.
    ///
    /// `data` must contain exactly `n*(n-1)/2` elements, stored as
    /// `data[row*(row-1)/2 + col]` where `row > col`.
    pub(crate) fn from_raw(n: usize, data: Vec<DtwDistance>) -> Self {
        debug_assert_eq!(data.len(), n * (n - 1) / 2);
        Self { n, data }
    }

    /// Return the number of series in the matrix.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return true if the matrix is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Return the distance between series `i` and series `j`.
    ///
    /// Returns zero distance for `i == j` (diagonal).
    ///
    /// # Panics
    ///
    /// Panics if `i >= n` or `j >= n`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> DtwDistance {
        assert!(i < self.n, "row index {i} out of bounds for matrix of size {}", self.n);
        assert!(j < self.n, "column index {j} out of bounds for matrix of size {}", self.n);
        if i == j {
            return DtwDistance::new(0.0);
        }
        let (row, col) = if i > j { (i, j) } else { (j, i) };
        self.data[row * (row - 1) / 2 + col]
    }

    /// Iterate over all unique pairs `(i, j, distance)` where `i > j`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, DtwDistance)> + '_ {
        (1..self.n).flat_map(move |i| {
            (0..i).map(move |j| (i, j, self.data[i * (i - 1) / 2 + j]))
        })
    }

    /// Return all distances from series `i` to every other series.
    ///
    /// Returns a vector of length `n` where index `j` is the distance from `i` to `j`.
    #[must_use]
    pub fn row(&self, i: usize) -> Vec<DtwDistance> {
        (0..self.n).map(|j| self.get(i, j)).collect()
    }
}

impl Index<(usize, usize)> for DistanceMatrix {
    type Output = DtwDistance;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        // Cannot return a reference to a computed value for the diagonal or transposed case.
        // We store lower-triangular, so we can only return a reference when row > col.
        // For the Index trait, we require i > j (strict lower triangle).
        assert!(i != j, "cannot index diagonal — use get() instead");
        let (row, col) = if i > j { (i, j) } else { (j, i) };
        &self.data[row * (row - 1) / 2 + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix() -> DistanceMatrix {
        // 4 series: 4*(4-1)/2 = 6 distances
        // Layout: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2)
        let data = vec![
            DtwDistance::new(1.0),
            DtwDistance::new(2.0),
            DtwDistance::new(3.0),
            DtwDistance::new(4.0),
            DtwDistance::new(5.0),
            DtwDistance::new(6.0),
        ];
        DistanceMatrix::from_raw(4, data)
    }

    #[test]
    fn diagonal_is_zero() {
        let m = make_matrix();
        for i in 0..4 {
            assert_eq!(m.get(i, i).value(), 0.0);
        }
    }

    #[test]
    fn symmetric_access() {
        let m = make_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m.get(i, j).value(), m.get(j, i).value());
            }
        }
    }

    #[test]
    fn specific_values() {
        let m = make_matrix();
        assert_eq!(m.get(1, 0).value(), 1.0);
        assert_eq!(m.get(2, 0).value(), 2.0);
        assert_eq!(m.get(2, 1).value(), 3.0);
        assert_eq!(m.get(3, 0).value(), 4.0);
        assert_eq!(m.get(3, 1).value(), 5.0);
        assert_eq!(m.get(3, 2).value(), 6.0);
    }

    #[test]
    fn index_trait() {
        let m = make_matrix();
        assert_eq!(m[(1, 0)].value(), 1.0);
        assert_eq!(m[(2, 1)].value(), 3.0);
    }

    #[test]
    fn iter_yields_lower_triangle() {
        let m = make_matrix();
        let pairs: Vec<_> = m.iter().collect();
        assert_eq!(pairs.len(), 6);
        assert_eq!(pairs[0], (1, 0, DtwDistance::new(1.0)));
        assert_eq!(pairs[5], (3, 2, DtwDistance::new(6.0)));
    }

    #[test]
    fn row_distances() {
        let m = make_matrix();
        let row0: Vec<f64> = m.row(0).iter().map(|d| d.value()).collect();
        assert_eq!(row0, vec![0.0, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn len_and_is_empty() {
        let m = make_matrix();
        assert_eq!(m.len(), 4);
        assert!(!m.is_empty());
    }
}
