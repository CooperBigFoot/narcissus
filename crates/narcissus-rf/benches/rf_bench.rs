//! Criterion benchmarks for narcissus-rf: Random Forest training and prediction.

use criterion::{Criterion, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use narcissus_rf::RandomForestConfig;

fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<String>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut features = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let class = i % n_classes;
        labels.push(class);
        let row: Vec<f64> = (0..n_features)
            .map(|f| {
                let base = if f < 3 { class as f64 * 3.0 } else { 0.0 };
                base + rng.r#gen::<f64>() * 0.5
            })
            .collect();
        features.push(row);
    }
    let names: Vec<String> = (0..n_features).map(|f| format!("f{f}")).collect();
    (features, labels, names)
}

fn bench_rf_train(c: &mut Criterion) {
    let (features, labels, names) = make_classification(500, 20, 5, 42);
    let cfg = RandomForestConfig::new(50).unwrap().with_seed(42);

    c.bench_function("rf_train_500x20_5class_50trees", |b| {
        b.iter(|| cfg.fit(&features, &labels, &names).unwrap());
    });
}

fn bench_rf_predict_batch(c: &mut Criterion) {
    let (features, labels, names) = make_classification(500, 20, 5, 42);
    let cfg = RandomForestConfig::new(50).unwrap().with_seed(42);
    let result = cfg.fit(&features, &labels, &names).unwrap();
    let forest = result.into_forest();

    c.bench_function("rf_predict_batch_500x20_50trees", |b| {
        b.iter(|| forest.predict_batch(&features).unwrap());
    });
}

fn bench_find_best_split(c: &mut Criterion) {
    // Proxy for split-finding: train a single-tree forest on 500 samples.
    let (features, labels, names) = make_classification(500, 20, 5, 42);
    let cfg = RandomForestConfig::new(1).unwrap().with_seed(42);

    c.bench_function("rf_single_tree_500x20_5class", |b| {
        b.iter(|| cfg.fit(&features, &labels, &names).unwrap());
    });
}

criterion_group!(benches, bench_rf_train, bench_rf_predict_batch, bench_find_best_split);
criterion_main!(benches);
