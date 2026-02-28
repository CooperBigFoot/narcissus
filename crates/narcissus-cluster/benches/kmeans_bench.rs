//! Criterion benchmarks for narcissus-cluster: K-means fit, assign, init, and optimize sweep.

use criterion::{Criterion, criterion_group, criterion_main};

use narcissus_cluster::{KMeansConfig, OptimizeConfig};
use narcissus_dtw::{BandConstraint, TimeSeries};

fn make_cluster_data() -> Vec<TimeSeries> {
    let offsets = [0.0, 5.0, 10.0, 15.0, 20.0];
    let mut series = Vec::new();
    for &offset in &offsets {
        for j in 0..20 {
            let values: Vec<f64> = (0..64)
                .map(|i| (i as f64 * 0.1).sin() + offset + j as f64 * 0.01)
                .collect();
            series.push(TimeSeries::new(values).unwrap());
        }
    }
    series
}

fn bench_kmeans_fit(c: &mut Criterion) {
    let series = make_cluster_data();
    let cfg = KMeansConfig::new(5, BandConstraint::SakoeChibaRadius(2))
        .unwrap()
        .with_n_init(3)
        .with_max_iter(20)
        .with_seed(42);

    c.bench_function("kmeans_fit_100x64_k5_ninit3", |b| {
        b.iter(|| cfg.fit(&series).unwrap());
    });
}

fn bench_kmeans_assign_only(c: &mut Criterion) {
    let series = make_cluster_data();
    // Approximate one assignment step by running n_init=1, max_iter=1 from a seeded start.
    let cfg = KMeansConfig::new(5, BandConstraint::SakoeChibaRadius(2))
        .unwrap()
        .with_n_init(1)
        .with_max_iter(1)
        .with_seed(42);

    c.bench_function("kmeans_assign_only_100x64_k5", |b| {
        b.iter(|| cfg.fit(&series).unwrap());
    });
}

fn bench_kmeans_plus_plus(c: &mut Criterion) {
    let series = make_cluster_data();
    // Isolate k-means++ initialization: n_init=1, max_iter=1.
    let cfg = KMeansConfig::new(5, BandConstraint::SakoeChibaRadius(2))
        .unwrap()
        .with_n_init(1)
        .with_max_iter(1)
        .with_seed(99);

    c.bench_function("kmeans_plus_plus_init_100x64_k5", |b| {
        b.iter(|| cfg.fit(&series).unwrap());
    });
}

fn bench_optimize_sweep(c: &mut Criterion) {
    let all_series = make_cluster_data();
    let series: Vec<TimeSeries> = all_series.into_iter().take(50).collect();
    let cfg = OptimizeConfig::new(2, 6, BandConstraint::SakoeChibaRadius(2))
        .unwrap()
        .with_n_init(2)
        .with_max_iter(10)
        .with_seed(42);

    c.bench_function("optimize_sweep_50x64_k2to6", |b| {
        b.iter(|| cfg.fit(&series).unwrap());
    });
}

criterion_group!(
    benches,
    bench_kmeans_fit,
    bench_kmeans_assign_only,
    bench_kmeans_plus_plus,
    bench_optimize_sweep
);
criterion_main!(benches);
