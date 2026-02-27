use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use tracing::{info, warn};

use narcissus_cluster::{KMeansConfig, OptimizeConfig};
use narcissus_dtw::BandConstraint;
use narcissus_io::{ExperimentName, ResultWriter, TimeSeriesReader};

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
    Evaluate,

    /// Predict cluster membership for new basins
    Predict,
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

fn build_constraint(warping_window: usize) -> BandConstraint {
    if warping_window == 0 {
        BandConstraint::Unconstrained
    } else {
        BandConstraint::SakoeChibaRadius(warping_window)
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
            tuning,
        } => {
            let constraint = build_constraint(tuning.warping_window);
            let experiment_name = ExperimentName::new(experiment.clone())?;

            // Read dataset
            let dataset = TimeSeriesReader::new(&data)
                .read()
                .context("failed to read input CSV")?;
            info!(n_basins = dataset.series.len(), "dataset loaded");

            // Build and run optimize
            let config = OptimizeConfig::new(min_k, max_k, constraint)?
                .with_n_init(tuning.n_init)
                .with_max_iter(tuning.max_iter)
                .with_tol(tuning.tol)
                .with_seed(cli.seed);

            let result = config
                .fit(&dataset.series)
                .context("optimization failed")?;

            // Write JSON artifact
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            writer.write_optimize(dataset.series.len(), &result)?;

            // Build and print stdout summary
            let output = OptimizeOutput {
                experiment,
                n_basins: dataset.series.len(),
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

            // Build and run clustering
            let config = KMeansConfig::new(k, constraint)?
                .with_n_init(tuning.n_init)
                .with_max_iter(tuning.max_iter)
                .with_tol(tuning.tol)
                .with_seed(cli.seed);

            let result = config
                .fit(&dataset.series)
                .context("clustering failed")?;

            // Write JSON artifact
            let writer = ResultWriter::new(&output_dir, experiment_name)?;
            writer.write_cluster(&dataset.basin_ids, &result)?;

            // Build and print stdout summary
            let output = ClusterOutput {
                experiment,
                k: result.centroids.len(),
                inertia: result.inertia.value(),
                n_basins: dataset.series.len(),
                converged: result.converged,
                cluster_sizes: result.cluster_sizes(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        }

        Command::Evaluate => {
            warn!("evaluate: not yet implemented");
        }

        Command::Predict => {
            warn!("predict: not yet implemented");
        }
    }

    Ok(())
}
