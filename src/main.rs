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
    /// Prune old or low-importance memories
    Prune {
        /// Database URL (required, or set MEMENTO_DATABASE_URL env var)
        #[arg(long)]
        database_url: Option<String>,
        /// Prune memories older than this (e.g., "30d", "6m", "1y" or "2024-01-01")
        #[arg(long)]
        older_than: Option<String>,
        /// Prune memories with importance below this threshold (0.0-1.0)
        #[arg(long)]
        importance_below: Option<f64>,
        /// Prune memories that have never been accessed
        #[arg(long)]
        never_accessed: bool,
        /// Filter by agent_id
        #[arg(long)]
        agent_id: Option<String>,
        /// Dry run - show what would be deleted without actually deleting
        #[arg(long)]
        dry_run: bool,
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
        Some(Commands::Prune { 
            database_url, 
            older_than, 
            importance_below, 
            never_accessed,
            agent_id,
            dry_run,
        }) => {
            run_prune(database_url, older_than, importance_below, never_accessed, agent_id, dry_run).await?;
        }
        None => {
            // Default: start MCP server with no overrides
            memento::mcp::start_mcp_server(None, None, None).await?;
        }
    }
    
    Ok(())
}

/// Parse a cutoff date from either:
/// - Relative duration: "30d", "6m", "1y" (days, weeks, months, years)
/// - Absolute timestamp: "2024-01-01" or "2024-01-01T00:00:00Z"
fn parse_cutoff_date(s: &str) -> Result<chrono::DateTime<chrono::Utc>> {
    let s = s.trim();
    
    // Try parsing as absolute timestamp first
    // Format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
    if s.contains('-') && s.len() >= 10 {
        // Try full ISO 8601 format
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
            return Ok(dt.with_timezone(&chrono::Utc));
        }
        // Try date-only format (YYYY-MM-DD)
        if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let datetime = date.and_hms_opt(0, 0, 0).unwrap();
            return Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(datetime, chrono::Utc));
        }
        // Try datetime without timezone (YYYY-MM-DD HH:MM:SS)
        if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            return Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(datetime, chrono::Utc));
        }
    }
    
    // Parse as relative duration
    let s_lower = s.to_lowercase();
    let (num_str, unit) = s_lower.split_at(s_lower.len().saturating_sub(1));
    let num: i64 = num_str.parse().map_err(|_| anyhow::anyhow!(
        "Invalid format '{}'. Use relative (30d, 6m, 1y) or absolute (2024-01-01)", s
    ))?;
    
    let duration = match unit {
        "d" => chrono::Duration::days(num),
        "w" => chrono::Duration::weeks(num),
        "m" => chrono::Duration::days(num * 30), // Approximate month
        "y" => chrono::Duration::days(num * 365), // Approximate year
        _ => return Err(anyhow::anyhow!(
            "Invalid duration unit '{}'. Use d (days), w (weeks), m (months), or y (years)", unit
        )),
    };
    
    Ok(chrono::Utc::now() - duration)
}

async fn run_prune(
    database_url: Option<String>,
    older_than: Option<String>,
    importance_below: Option<f64>,
    never_accessed: bool,
    agent_id: Option<String>,
    dry_run: bool,
) -> Result<()> {
    use memento::DatabaseClient;
    
    // Get database URL from arg or environment
    let database_url = database_url
        .or_else(|| std::env::var("MEMENTO_DATABASE_URL").ok())
        .ok_or_else(|| anyhow::anyhow!("Database URL required. Use --database-url or set MEMENTO_DATABASE_URL env var"))?;
    
    // Need a minimal embedding dimension for schema init (won't actually use embeddings)
    let db = DatabaseClient::new(&database_url, 384).await?;
    
    // Build the prune criteria
    let cutoff_date = if let Some(date_str) = older_than {
        Some(parse_cutoff_date(&date_str)?)
    } else {
        None
    };
    
    // Count and optionally delete
    let count = db.prune_memories(
        cutoff_date,
        importance_below,
        never_accessed,
        agent_id.as_deref(),
        dry_run,
    ).await?;
    
    if dry_run {
        println!("Would prune {} memories (dry run)", count);
    } else {
        println!("Pruned {} memories", count);
    }
    
    Ok(())
}
