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
    Mcp,
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
        Some(Commands::Mcp) => {
            memento::mcp::start_mcp_server().await?;
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

