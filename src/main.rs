use anyhow::Result;
use clap::{Parser, Subcommand};
use memento::config::Config;
use memento::database::DatabaseClient;
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
    /// Start the REST API server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8000")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
    },
    /// Start the MCP server (stdio)
    Mcp {
        /// Database type: sqlite or postgresql (overrides MEMENTO_DATABASE_TYPE env var)
        #[arg(long)]
        database_type: Option<String>,
        /// Database URL (overrides MEMENTO_DATABASE_URL or DATABASE_URL env vars)
        #[arg(long)]
        database_url: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    
    let cli = Cli::parse();
    
    match cli.command {
        Some(Commands::Serve { port, host }) => {
            let config = Config::from_env();
            let db = DatabaseClient::new(&config.database_url).await?;
            start_server(db, host, port).await?;
        }
        Some(Commands::Mcp { database_type, database_url }) => {
            memento::mcp::start_mcp_server(database_type, database_url).await?;
        }
        None => {
            // Default to serve
            let config = Config::from_env();
            let db = DatabaseClient::new(&config.database_url).await?;
            start_server(db, "0.0.0.0".to_string(), 8000).await?;
        }
    }
    
    Ok(())
}

