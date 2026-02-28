//! Criterion benchmarks for narcissus-dtw: DTW distance, pairwise matrix, and DBA averaging.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use narcissus_dtw::{BandConstraint, DbaConfig, Dtw, TimeSeries};

fn make_sine_series(n: usize, offset: f64) -> TimeSeries {
    let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + offset).collect();
    TimeSeries::new(values).unwrap()
}

fn bench_dtw_distance(c: &mut Criterion) {
    let lengths = [64usize, 256, 1024];
    let bands: &[(Option<usize>, &str)] = &[
        (None, "unconstrained"),
        (Some(2), "band_r2"),
        (Some(10), "band_r10"),
    ];

    let mut group = c.benchmark_group("dtw_distance");

    for &len in &lengths {
        for &(band, band_label) in bands {
            let id = BenchmarkId::new(format!("len{len}"), band_label);
            let a = make_sine_series(len, 0.0);
            let b = make_sine_series(len, 1.0);
            let dtw = match band {
                None => Dtw::unconstrained(),
                Some(r) => Dtw::with_sakoe_chiba(r),
            };

            group.bench_with_input(id, &(a, b, dtw), |bencher, (a, b, dtw)| {
                bencher.iter(|| dtw.distance(a.as_view(), b.as_view()));
            });
        }
    }

    group.finish();
}

fn bench_dtw_pairwise(c: &mut Criterion) {
    let series: Vec<TimeSeries> = (0..50)
        .map(|i| make_sine_series(128, i as f64 * 0.2))
        .collect();
    let dtw = Dtw::with_sakoe_chiba(2);

    c.bench_function("dtw_pairwise_50x128_r2", |b| {
        b.iter(|| dtw.pairwise(&series));
    });
}

fn bench_dba_average(c: &mut Criterion) {
    let series: Vec<TimeSeries> = (0..20)
        .map(|i| make_sine_series(128, i as f64 * 0.1))
        .collect();
    let views: Vec<_> = series.iter().map(|s| s.as_view()).collect();
    let config = DbaConfig::new(BandConstraint::SakoeChibaRadius(2)).with_max_iter(10);

    c.bench_function("dba_average_20x128_r2_iter10", |b| {
        b.iter(|| config.average(&views).unwrap());
    });
}

criterion_group!(benches, bench_dtw_distance, bench_dtw_pairwise, bench_dba_average);
criterion_main!(benches);
