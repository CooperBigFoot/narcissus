use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use tracing::info;

use narcissus_cluster::{compute_silhouette, InitStrategy, KMeansConfig, OptimizeConfig};
use narcissus_dtw::BandConstraint;
use narcissus_io::{align, AttributeReader, ExperimentName, ResultWriter, TimeSeriesReader};
use narcissus_rf::{CrossValidation, OobMode, RandomForest, RandomForestConfig, SplitMethod};

#[derive(Parser)]
#[command(name = "narcissus")]
#[command(about = "Shape-based hydrological basin clustering and classification")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// RNG seed for reproducibility
    #[arg(long, default_value_t = 42, global = true)]
    seed: u64,

    /// Enable verbose (debug-level) logging
    #[arg(long, global = true)]
    verbose: bool,

    /// Suppress all output except errors
    #[arg(long, global = true)]
    quiet: bool,

    /// Number of threads for parallel computation (defaults to all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
}

/// Shared tuning parameters for clustering algorithms.
#[derive(Args, Debug, Clone)]
struct TuningArgs {
    /// Number of independent K-means restarts (best result kept)
    #[arg(long, default_value_t = 10)]
    n_init: usize,

    /// Maximum iterations per K-means run
    #[arg(long, default_value_t = 75)]
    max_iter: usize,

    /// Sakoe-Chiba warping window radius (0 = unconstrained)
    #[arg(long, default_value_t = 2)]
    warping_window: usize,

    /// Convergence tolerance for inertia change
    #[arg(long, default_value_t = 1e-4)]
    tol: f64,

    /// Z-normalize time series before DTW computation
    #[arg(long, default_value_t = false)]
    normalize: bool,

    /// Use derivative DTW (Keogh-Pazzani first derivative)
    #[arg(long, default_value_t = false)]
    derivative: bool,

    /// DBA initialization strategy: "mean" or "medoid"
    #[arg(long, default_value = "mean")]
    dba_init: String,

    /// Fraction of cluster members to sample per DBA iteration (1.0 = full DBA)
    #[arg(long, default_value_t = 1.0)]
    dba_sample_fraction: f64,

    /// Centroid averaging method: "dba" or "ssg"
    #[arg(long, default_value = "dba")]
    centroid_method: String,

    /// Use Elkan's triangle inequality acceleration for K-means
    #[arg(long, default_value_t = false)]
    elkan: bool,

    /// Initialization strategy: "kpp" (K-Means++) or "parallel" (K-Means‖)
    #[arg(long, default_value = "kpp")]
    init_strategy: String,
}

#[derive(Subcommand)]
enum Command {
    /// Run clustering for a range of k values to find the optimal cluster count
    Optimize {
        /// Path to the input CSV file
        #[arg(long)]
        data: PathBuf,

        /// Minimum number of clusters to try
        #[arg(long)]
        min_k: usize,

        /// Maximum number of clusters to try
        #[arg(long)]
        max_k: usize,

        /// Experiment name for output files (must match [a-zA-Z0-9_-]+)
        #[arg(long)]
        experiment: String,

        /// Output directory for result files
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,

        /// Compute silhouette scores alongside inertia
        #[arg(long, default_value_t = false)]
        silhouette: bool,

        #[command(flatten)]
        tuning: TuningArgs,
    },

    /// Cluster basins into k groups using DTW K-means
    Cluster {
        /// Path to the input CSV file
        #[arg(long)]
        data: PathBuf,

        /// Number of clusters
        #[arg(long)]
        k: usize,

        /// Experiment name for output files (must match [a-zA-Z0-9_-]+)
        #[arg(long)]
        experiment: String,

        /// Output directory for result files
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,

        #[command(flatten)]
        tuning: TuningArgs,
    },

    /// Train and evaluate a classifier on cluster labels + basin attributes
    Evaluate {
        /// Path to the time series CSV file
        #[arg(long)]
        data: PathBuf,

        /// Path to the basin attributes CSV file
        #[arg(long)]
        attributes: PathBuf,

        /// Number of clusters
        #[arg(long)]
        k: usize,

        /// Experiment name for output files
        #[arg(long)]
        experiment: String,

        /// Output directory for result files
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,

        /// Number of cross-validation folds
        #[arg(long, default_value_t = 5)]
        cv_folds: usize,

        /// Number of trees in the Random Forest
        #[arg(long, default_value_t = 100)]
        n_trees: usize,

        /// Maximum tree depth (unlimited if not set)
        #[arg(long)]
        max_depth: Option<usize>,

        /// Split-finding strategy: "exact", "extra-trees", or "histogram"
        #[arg(long, default_value = "exact")]
        split_method: String,

        #[command(flatten)]
        tuning: TuningArgs,
    },

    /// Predict cluster membership for new basins
    Predict {
        /// Path to the trained model binary
        #[arg(long)]
        model: PathBuf,

        /// Path to the basin attributes CSV file
        #[arg(long)]
        attributes: PathBuf,

        /// Number of top-k classes to output per basin
        #[arg(long, default_value_t = 3)]
        top_k: usize,

        /// Experiment name for output files
        #[arg(long)]
        experiment: String,

        /// Output directory for result files
        #[arg(long, default_value = ".")]
        output_dir: PathBuf,
    },
}

// --- JSON stdout output structs ---

#[derive(Serialize)]
struct OptimizeOutput {
    experiment: String,
    n_basins: usize,
    best_k: Option<usize>,
    results: Vec<KResultOutput>,
}

#[derive(Serialize)]
struct KResultOutput {
    k: usize,
    inertia: f64,
}

#[derive(Serialize)]
struct ClusterOutput {
    experiment: String,
    k: usize,
    inertia: f64,
    n_basins: usize,
    converged: bool,
    cluster_sizes: Vec<usize>,
}

#[derive(Serialize)]
struct EvaluateOutput {
    experiment: String,
    n_basins: usize,
    k: usize,
    cv_mean_accuracy: f64,
    cv_std_accuracy: f64,
    oob_accuracy: Option<f64>,
    n_trees: usize,
    n_features: usize,
}

#[derive(Serialize)]
struct PredictOutput {
    experiment: String,
    n_basins: usize,
    model_n_trees: usize,
    model_n_features: usize,
    model_n_classes: usize,
}

fn build_constraint(warping_window: usize) -> BandConstraint {
    if warping_window == 0 {
        BandConstraint::Unconstrained
    } else {
        BandConstraint::SakoeChibaRadius(warping_window)
    }
}

fn preprocess_series(
    series: Vec<narcissus_dtw::TimeSeries>,
    normalize: bool,
    use_derivative: bool,
) -> Result<Vec<narcissus_dtw::TimeSeries>> {
    let mut result = series;
    if normalize {
        result = narcissus_dtw::z_normalize_batch(&result)
            .context("z-normalization failed")?;
        info!(n = result.len(), "z-normalized series");
    }
    if use_derivative {
        result = result
            .into_iter()
            .map(|s| narcissus_dtw::derivative(&s).context("derivative computation failed"))
            .collect::<Result<Vec<_>>>()?;
        info!(n = result.len(), "computed derivative series");
    }
    Ok(result)
}

fn parse_split_method(s: &str) -> Result<SplitMethod> {
    match s {
        "exact" => Ok(SplitMethod::Exact),
        "extra-trees" => Ok(SplitMethod::ExtraTrees),
        "histogram" => Ok(SplitMethod::Histogram { n_bins: 256 }),
        other => anyhow::bail!("unknown split method: {other} (expected exact, extra-trees, or histogram)"),
    }
}

fn parse_init_strategy(s: &str) -> Result<InitStrategy> {
    match s {
        "kpp" => Ok(InitStrategy::KMeansPlusPlus),
        "parallel" => Ok(InitStrategy::KMeansParallel { oversample_factor: 2.0 }),
        other => anyhow::bail!("unknown init strategy: {other} (expected kpp or parallel)"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let filter = match (cli.verbose, cli.quiet) {
        (true, _) => "debug",
        (_, true) => "error",
        _ => "info",
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    // Configure Rayon thread pool
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to configure thread pool")?;
        info!(threads, "thread pool configured");
    }

    match cli.command {
        Command::Optimize {
            data,
            min_k,
            max_k,
            experiment,
            output_dir,
            silhouette,
            tuning,
        } => {
            let constraint = build_constraint(tuning.warping_window);
            let experiment_name = ExperimentName::new(experiment.clone())?;

            // Read dataset
            let dataset = TimeSeriesReader::new(&data)
                .read()
                .context("failed to read input CSV")?;
            info!(n_basins = dataset.series.len(), "dataset loaded");

            // Preprocess series
            let series = preprocess_series(dataset.series, tuning.normalize, tuning.derivative)?;

            // Build and run optimize
            let config = OptimizeConfig::new(min_k, max_k, constraint)?
                .with_n_init(tuning.n_init)
                .with_max_iter(tuning.max_iter)
                .with_tol(tuning.tol)
                .with_seed(cli.seed);

            let result = config
                .fit(&series)
                .context("optimization failed")?;

            // Optionally compute silhouette for the best k
            if silhouette
                && let Some(best_k_val) = result.best_k()
            {
                    let cluster_config = KMeansConfig::new(best_k_val, constraint)?
                        .with_n_init(tuning.n_init)
                        .with_max_iter(tuning.max_iter)
                        .with_tol(tuning.tol)
                        .with_seed(cli.seed);
                    let cluster_result = cluster_config.fit(&series)?;
                    let dtw = if tuning.warping_window == 0 {
                        narcissus_dtw::Dtw::unconstrained()
                    } else {
                        narcissus_dtw::Dtw::with_sakoe_chiba(tuning.warping_window)
                    };
                    let sil = compute_silhouette(
                        &series,
                        &cluster_result.assignments,
                        best_k_val,
                        &dtw,
                    )?;
                    info!(silhouette_score = sil.mean_score, "silhouette computed");
            }

            // Write JSON artifact
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            writer.write_optimize(series.len(), &result)?;

            // Build and print stdout summary
            let output = OptimizeOutput {
                experiment,
                n_basins: series.len(),
                best_k: result.best_k(),
                results: result
                    .results
                    .iter()
                    .map(|r| KResultOutput {
                        k: r.k,
                        inertia: r.inertia.value(),
                    })
                    .collect(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }

        Command::Cluster {
            data,
            k,
            experiment,
            output_dir,
            tuning,
        } => {
            let constraint = build_constraint(tuning.warping_window);
            let experiment_name = ExperimentName::new(experiment.clone())?;

            // Read dataset
            let dataset = TimeSeriesReader::new(&data)
                .read()
                .context("failed to read input CSV")?;
            info!(n_basins = dataset.series.len(), "dataset loaded");

            let basin_ids = dataset.basin_ids;

            // Preprocess series
            let series = preprocess_series(dataset.series, tuning.normalize, tuning.derivative)?;

            // Parse init strategy
            let init_strategy = parse_init_strategy(&tuning.init_strategy)?;

            // Build and run clustering
            let config = KMeansConfig::new(k, constraint)?
                .with_n_init(tuning.n_init)
                .with_max_iter(tuning.max_iter)
                .with_tol(tuning.tol)
                .with_seed(cli.seed)
                .with_use_elkan(tuning.elkan)
                .with_init_strategy(init_strategy);

            let result = config
                .fit(&series)
                .context("clustering failed")?;

            // Write JSON artifact
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            writer.write_cluster(&basin_ids, &result)?;

            // Build and print stdout summary
            let output = ClusterOutput {
                experiment,
                k: result.centroids.len(),
                inertia: result.inertia.value(),
                n_basins: series.len(),
                converged: result.converged,
                cluster_sizes: result.cluster_sizes(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }

        Command::Evaluate {
            data,
            attributes,
            k,
            experiment,
            output_dir,
            cv_folds,
            n_trees,
            max_depth,
            split_method,
            tuning,
        } => {
            let constraint = build_constraint(tuning.warping_window);
            let experiment_name = ExperimentName::new(experiment.clone())?;

            // 1. Read time series and preprocess
            let dataset = TimeSeriesReader::new(&data)
                .read()
                .context("failed to read time series CSV")?;
            info!(n_basins = dataset.series.len(), "time series loaded");

            let basin_ids = dataset.basin_ids;
            let series = preprocess_series(dataset.series, tuning.normalize, tuning.derivative)?;

            // Parse init strategy
            let init_strategy = parse_init_strategy(&tuning.init_strategy)?;

            // 2. Cluster
            let cluster_config = KMeansConfig::new(k, constraint)?
                .with_n_init(tuning.n_init)
                .with_max_iter(tuning.max_iter)
                .with_tol(tuning.tol)
                .with_seed(cli.seed)
                .with_use_elkan(tuning.elkan)
                .with_init_strategy(init_strategy);
            let cluster_result = cluster_config
                .fit(&series)
                .context("clustering failed")?;
            info!(k, inertia = cluster_result.inertia.value(), "clustering complete");

            // 3. Read attributes and align
            let attr_dataset = AttributeReader::new(&attributes)
                .read()
                .context("failed to read attributes CSV")?;

            let cluster_labels: Vec<usize> = cluster_result
                .assignments
                .iter()
                .map(|l| l.index())
                .collect();

            let aligned = align(&basin_ids, &cluster_labels, &attr_dataset)
                .context("failed to align attributes with cluster assignments")?;
            info!(n_aligned = aligned.n_samples(), "alignment complete");

            // 4. Cross-validate
            let feature_names: Vec<String> = aligned.feature_names().to_vec();
            let split_method_parsed = parse_split_method(&split_method)?;
            let rf_config = RandomForestConfig::new(n_trees)?
                .with_max_depth(max_depth)
                .with_split_method(split_method_parsed)
                .with_seed(cli.seed);

            let cv = CrossValidation::new(cv_folds)?
                .with_seed(cli.seed);
            let cv_result = cv
                .evaluate(&rf_config, aligned.features(), aligned.labels(), &feature_names)
                .context("cross-validation failed")?;
            info!(
                mean_accuracy = cv_result.mean_accuracy,
                std_accuracy = cv_result.std_accuracy,
                "cross-validation complete"
            );

            // 5. Train final model on all data with OOB
            let final_config = RandomForestConfig::new(n_trees)?
                .with_max_depth(max_depth)
                .with_split_method(split_method_parsed)
                .with_oob_mode(OobMode::Enabled)
                .with_seed(cli.seed);
            let train_result = final_config
                .fit(aligned.features(), aligned.labels(), &feature_names)
                .context("final model training failed")?;

            let oob_accuracy = train_result.oob_score().map(|s| s.accuracy);
            info!(oob_accuracy = ?oob_accuracy, "final model trained");

            // 6. Save model
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            train_result
                .forest()
                .save(writer.model_path())
                .context("failed to save model")?;
            info!(path = %writer.model_path().display(), "model saved");

            // 7. Write evaluation JSON
            let importances: Vec<f64> = cv_result.feature_importances.iter().map(|f| f.importance).collect();
            let ranks: Vec<usize> = cv_result.feature_importances.iter().map(|f| f.rank).collect();
            let imp_names: Vec<String> = cv_result.feature_importances.iter().map(|f| f.name.clone()).collect();
            let class_metrics: Vec<(f64, f64, f64, usize)> = cv_result
                .confusion_matrix
                .class_metrics()
                .iter()
                .map(|m| (m.precision, m.recall, m.f1, m.support))
                .collect();

            writer.write_evaluation(
                cv_result.mean_accuracy,
                cv_result.std_accuracy,
                &cv_result.fold_accuracies,
                oob_accuracy,
                &imp_names,
                &importances,
                &ranks,
                cv_result.confusion_matrix.as_rows(),
                cv_result.n_classes,
                &class_metrics,
            )?;

            // 8. Write cluster JSON
            writer.write_cluster(&basin_ids, &cluster_result)?;

            // 9. Print summary
            let output = EvaluateOutput {
                experiment,
                n_basins: aligned.n_samples(),
                k,
                cv_mean_accuracy: cv_result.mean_accuracy,
                cv_std_accuracy: cv_result.std_accuracy,
                oob_accuracy,
                n_trees,
                n_features: aligned.n_features(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }

        Command::Predict {
            model,
            attributes,
            top_k,
            experiment,
            output_dir,
        } => {
            let experiment_name = ExperimentName::new(experiment.clone())?;

            // 1. Load model
            let forest = RandomForest::load(&model)
                .context("failed to load model")?;
            info!(
                n_trees = forest.n_trees(),
                n_features = forest.n_features(),
                n_classes = forest.n_classes(),
                "model loaded"
            );

            // 2. Read attributes
            let attr_dataset = AttributeReader::new(&attributes)
                .read()
                .context("failed to read attributes CSV")?;
            info!(n_basins = attr_dataset.n_samples(), "attributes loaded");

            // 3. Predict
            let proba_results = forest
                .predict_proba_batch(attr_dataset.features())
                .context("prediction failed")?;

            // 4. Build prediction entries
            let predictions: Vec<(String, Vec<(usize, f64)>)> = attr_dataset
                .basin_ids()
                .iter()
                .zip(proba_results.iter())
                .map(|(basin_id, dist)| {
                    (basin_id.as_str().to_string(), dist.top_k(top_k))
                })
                .collect();

            // 5. Write predictions JSON
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            writer.write_predictions(&predictions)?;

            // 6. Print summary
            let output = PredictOutput {
                experiment,
                n_basins: attr_dataset.n_samples(),
                model_n_trees: forest.n_trees(),
                model_n_features: forest.n_features(),
                model_n_classes: forest.n_classes(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}
