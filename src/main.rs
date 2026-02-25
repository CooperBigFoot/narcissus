use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

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
}

#[derive(Subcommand)]
enum Command {
    /// Run clustering for a range of k values to find the optimal cluster count
    Optimize,
    /// Cluster basins into k groups using DTW K-means
    Cluster,
    /// Train and evaluate a classifier on cluster labels + basin attributes
    Evaluate,
    /// Predict cluster membership for new basins
    Predict,
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

    match cli.command {
        Command::Optimize => {
            info!("optimize: not yet implemented");
        }
        Command::Cluster => {
            info!("cluster: not yet implemented");
        }
        Command::Evaluate => {
            info!("evaluate: not yet implemented");
        }
        Command::Predict => {
            info!("predict: not yet implemented");
        }
    }

    Ok(())
}
