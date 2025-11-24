use anyhow::Result;
use clap::{Parser, Subcommand};
use memento::server::start_server;

#[derive(Parser)]
#[command(name = "memento")]
#[command(about = "Universal Agent Memory Engine")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Memento configuration (interactive setup)
    Init,
    /// Start the REST API server
    Serve {
        /// Port to listen on (overrides PORT env var)
        #[arg(short, long)]
        port: Option<u16>,
        /// Host to bind to (overrides HOST env var)
        #[arg(long)]
        host: Option<String>,
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
    /// Start the MCP server (stdio)
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
        Some(Commands::Serve { port, host, database_type, database_url, embedding_dim }) => {
            let (config, db, embedding_provider) = memento::bootstrap::init_database_and_provider(
                database_type,
                database_url,
                embedding_dim,
            ).await?;
            let final_port = port.unwrap_or(config.port);
            let final_host = host.unwrap_or(config.host);
            start_server(db, embedding_provider, final_host, final_port).await?;
        }
        Some(Commands::Mcp { database_type, database_url, embedding_dim }) => {
            memento::mcp::start_mcp_server(database_type, database_url, embedding_dim).await?;
        }
        None => {
            let (config, db, embedding_provider) = memento::bootstrap::init_database_and_provider(
                None,
                None,
                None,
            ).await?;
            start_server(db, embedding_provider, config.host, config.port).await?;
        }
    }
    
    Ok(())
}

