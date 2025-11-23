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
        Some(Commands::Serve { port, host, database_type, database_url }) => {
            let mut config = Config::from_env();
            
            if let Some(db_type) = database_type {
                let db_url = database_url
                    .or_else(|| std::env::var("MEMENTO_DATABASE_URL").ok())
                    .or_else(|| {
                        match db_type.as_str() {
                            "postgresql" | "postgres" => std::env::var("DATABASE_URL").ok(),
                            _ => Some("./memento.db".to_string()),
                        }
                    });
                
                let final_db_url = match db_type.as_str() {
                    "postgresql" | "postgres" => {
                        let url = db_url.ok_or_else(|| {
                            anyhow::anyhow!("PostgreSQL requires database URL")
                        })?;
                        if !url.starts_with("postgresql://") && !url.starts_with("postgres://") {
                            format!("postgresql://{}", url)
                        } else {
                            url
                        }
                    }
                    _ => {
                        let path = db_url.unwrap_or_else(|| "./memento.db".to_string());
                        let path = path.strip_prefix("sqlite://").unwrap_or(&path);
                        format!("sqlite://{}", path)
                    }
                };
                config.database_url = final_db_url;
            } else if let Some(db_url) = database_url {
                config.database_url = db_url;
            }
            
            let db = DatabaseClient::new(&config.database_url).await?;
            let final_port = port.unwrap_or(config.port);
            let final_host = host.unwrap_or(config.host);
            start_server(db, final_host, final_port).await?;
        }
        Some(Commands::Mcp { database_type, database_url }) => {
            memento::mcp::start_mcp_server(database_type, database_url).await?;
        }
        None => {
            let config = Config::from_env();
            let db = DatabaseClient::new(&config.database_url).await?;
            start_server(db, config.host, config.port).await?;
        }
    }
    
    Ok(())
}

