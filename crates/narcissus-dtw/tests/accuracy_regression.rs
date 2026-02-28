//! Accuracy regression tests for narcissus-dtw.
//!
//! These tests verify that algorithmic changes do not degrade DTW distance accuracy
//! or DBA barycenter quality. Reference values were computed from the implementation
//! and are hardcoded to catch regressions.

use narcissus_dtw::{BandConstraint, DbaConfig, Dtw, TimeSeries};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn ts(values: Vec<f64>) -> TimeSeries {
    TimeSeries::new(values).expect("valid test series")
}

// ---------------------------------------------------------------------------
// a) dtw_distances_match_known_values
// ---------------------------------------------------------------------------

/// Verify DTW distances for 10 synthetic series pairs match hardcoded reference values.
///
/// Reference values were obtained by running the implementation and capturing output.
#[test]
fn dtw_distances_match_known_values() {
    let pairs: Vec<(TimeSeries, TimeSeries)> = vec![
        (ts(vec![0.0, 0.0, 0.0]), ts(vec![1.0, 1.0, 1.0])),           // constant offset
        (ts(vec![0.0, 1.0, 0.0]), ts(vec![0.0, 0.0, 0.0])),           // single peak
        (ts(vec![1.0, 2.0, 3.0, 4.0]), ts(vec![1.0, 2.0, 3.0, 4.0])), // identical
        (ts(vec![1.0, 2.0, 3.0]), ts(vec![3.0, 2.0, 1.0])),           // reversed
        (ts(vec![0.0, 5.0, 0.0, 5.0]), ts(vec![5.0, 0.0, 5.0, 0.0])), // alternating
        (ts(vec![1.0]), ts(vec![5.0])),                                  // single point
        (ts(vec![0.0, 0.0, 1.0]), ts(vec![1.0, 0.0, 0.0])),           // shifted peak
        (ts(vec![0.0, 1.0, 2.0, 3.0, 4.0]), ts(vec![0.0, 0.0, 0.0, 0.0, 4.0])), // late ramp
        (ts(vec![10.0, 10.0, 10.0]), ts(vec![10.1, 9.9, 10.0])),       // tiny perturbation
        (ts(vec![0.0, 3.0, 0.0, 3.0, 0.0]), ts(vec![3.0, 0.0, 3.0, 0.0, 3.0])), // opposite phase
    ];

    // Reference distances (computed by running the implementation).
    let expected: Vec<f64> = vec![
        1.7320508075688772,    // [0,0,0] vs [1,1,1]
        1.0,                   // [0,1,0] vs [0,0,0]
        0.0,                   // identical
        2.8284271247461903,    // [1,2,3] vs [3,2,1] — DTW warps to minimize cost
        7.0710678118654755,    // alternating — DTW warps to minimize cost
        4.0,                   // [1] vs [5]
        1.4142135623730951,    // shifted peak
        2.449489742783178,     // late ramp
        0.14142135623730953,   // tiny perturbation
        4.242640687119285,     // opposite phase
    ];

    let dtw = Dtw::unconstrained();
    for (i, ((a, b), &exp)) in pairs.iter().zip(expected.iter()).enumerate() {
        let dist = dtw.distance(a.as_view(), b.as_view()).value();
        assert!(
            (dist - exp).abs() < 1e-10,
            "pair {i}: got {dist:.15}, expected {exp:.15}"
        );
    }
}

// ---------------------------------------------------------------------------
// b) dtw_distance_with_band_geq_unconstrained
// ---------------------------------------------------------------------------

/// Banded DTW distance must be >= unconstrained DTW distance for 5 pairs.
#[test]
fn dtw_distance_with_band_geq_unconstrained() {
    let pairs: Vec<(TimeSeries, TimeSeries)> = vec![
        (ts(vec![0.0, 1.0, 2.0, 3.0]), ts(vec![3.0, 2.0, 1.0, 0.0])),
        (ts(vec![1.0, 5.0, 1.0, 5.0, 1.0]), ts(vec![5.0, 1.0, 5.0, 1.0, 5.0])),
        (ts(vec![0.0, 0.0, 0.0, 1.0]), ts(vec![1.0, 0.0, 0.0, 0.0])),
        (ts(vec![1.0, 2.0, 3.0, 4.0, 5.0]), ts(vec![5.0, 4.0, 3.0, 2.0, 1.0])),
        (ts(vec![10.0, 0.0, 10.0]), ts(vec![0.0, 10.0, 0.0])),
    ];

    let unconstrained = Dtw::unconstrained();
    let banded = Dtw::with_sakoe_chiba(1);

    for (i, (a, b)) in pairs.iter().enumerate() {
        let d_unconstrained = unconstrained.distance(a.as_view(), b.as_view()).value();
        let d_banded = banded.distance(a.as_view(), b.as_view()).value();
        assert!(
            d_banded >= d_unconstrained - 1e-10,
            "pair {i}: banded {d_banded} < unconstrained {d_unconstrained}"
        );
    }
}

// ---------------------------------------------------------------------------
// c) dba_identical_series_equals_input
// ---------------------------------------------------------------------------

/// DBA centroid of 3 identical series must equal that series within 1e-10.
#[test]
fn dba_identical_series_equals_input() {
    let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let s1 = ts(values.clone());
    let s2 = ts(values.clone());
    let s3 = ts(values.clone());

    let views = [s1.as_view(), s2.as_view(), s3.as_view()];
    let config = DbaConfig::new(BandConstraint::Unconstrained);
    let result = config.average(&views).expect("DBA should succeed");

    let centroid = result.centroid.as_ref();
    assert_eq!(centroid.len(), values.len(), "centroid length mismatch");
    for (i, (&c, &v)) in centroid.iter().zip(values.iter()).enumerate() {
        assert!(
            (c - v).abs() < 1e-10,
            "centroid[{i}] = {c}, expected {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// d) dba_centroid_finite_and_correct_length
// ---------------------------------------------------------------------------

/// DBA of the archetype 9-series dataset must produce a finite centroid of correct length.
#[test]
fn dba_centroid_finite_and_correct_length() {
    let series: Vec<TimeSeries> = vec![
        ts(vec![0.0, 0.0, 0.0, 0.0]),
        ts(vec![0.1, 0.0, 0.0, 0.0]),
        ts(vec![0.0, 0.1, 0.0, 0.0]),
        ts(vec![5.0, 5.0, 5.0, 5.0]),
        ts(vec![5.1, 5.0, 5.0, 5.0]),
        ts(vec![5.0, 5.1, 5.0, 5.0]),
        ts(vec![10.0, 10.0, 10.0, 10.0]),
        ts(vec![10.1, 10.0, 10.0, 10.0]),
        ts(vec![10.0, 10.1, 10.0, 10.0]),
    ];

    let views: Vec<_> = series.iter().map(|s| s.as_view()).collect();
    let config = DbaConfig::new(BandConstraint::Unconstrained);
    let result = config.average(&views).expect("DBA should succeed");

    let centroid = result.centroid.as_ref();
    assert_eq!(centroid.len(), 4, "centroid length must match series length");
    for (i, &v) in centroid.iter().enumerate() {
        assert!(v.is_finite(), "centroid[{i}] is not finite: {v}");
    }

    // The mean of all 9 series is approx 5.0 per element.
    // DBA centroid should be close to the element-wise mean (~5.0).
    for (i, &v) in centroid.iter().enumerate() {
        assert!(
            (v - 5.0).abs() < 1.0,
            "centroid[{i}] = {v}, expected near 5.0"
        );
    }
}

// ---------------------------------------------------------------------------
// e) dtw_rolling_matches_full_matrix
// ---------------------------------------------------------------------------

/// `distance()` (rolling buffer) must match `distance_and_path().0` (full matrix) within 1e-10.
#[test]
fn dtw_rolling_matches_full_matrix() {
    let pairs: Vec<(TimeSeries, TimeSeries)> = vec![
        (ts(vec![1.0, 2.0, 3.0]), ts(vec![3.0, 2.0, 1.0])),
        (ts(vec![0.0, 5.0, 0.0, 5.0]), ts(vec![5.0, 0.0, 5.0, 0.0])),
        (ts(vec![1.0, 1.0, 1.0, 1.0, 1.0]), ts(vec![2.0, 2.0, 2.0, 2.0, 2.0])),
        (ts(vec![0.0, 1.0, 4.0, 9.0]), ts(vec![0.0, 2.0, 3.0, 8.0])),
        (ts(vec![10.0, 5.0, 1.0]), ts(vec![1.0, 5.0, 10.0])),
    ];

    let dtw = Dtw::unconstrained();
    for (i, (a, b)) in pairs.iter().enumerate() {
        let d_rolling = dtw.distance(a.as_view(), b.as_view()).value();
        let (d_full, _) = dtw.distance_and_path(a.as_view(), b.as_view());
        let d_full = d_full.value();
        assert!(
            (d_rolling - d_full).abs() < 1e-10,
            "pair {i}: rolling {d_rolling:.15} != full_matrix {d_full:.15}"
        );
    }
}
