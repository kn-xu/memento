use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "memento")]
#[command(about = "Universal Agent Memory Engine - MCP Server")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Memento configuration (interactive setup)
    Init,
    /// Start the MCP server (stdio) - this is the default command
    Mcp {
        /// Database type: sqlite or postgresql (overrides MEMENTO_DATABASE_TYPE env var)
        #[arg(long)]
        database_type: Option<String>,
        /// Database URL (overrides MEMENTO_DATABASE_URL or DATABASE_URL env vars)
        #[arg(long)]
        database_url: Option<String>,
        /// Embedding dimension override (overrides MEMENTO_EMBEDDING_DIM env var)
        #[arg(long)]
        embedding_dim: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    
    let cli = Cli::parse();
    
    match cli.command {
        Some(Commands::Init) => {
            memento::mcp::run_init()?;
        }
        Some(Commands::Mcp { database_type, database_url, embedding_dim }) => {
            memento::mcp::start_mcp_server(database_type, database_url, embedding_dim).await?;
        }
        None => {
            // Default: start MCP server with no overrides
            memento::mcp::start_mcp_server(None, None, None).await?;
        }
    }
    
    Ok(())
}
