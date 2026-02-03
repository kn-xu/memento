//! MCP (Model Context Protocol) server implementation.
//!
//! Handles JSON-RPC requests over stdio for tool invocations like
//! memento.store, memento.search, memento.forget, etc.

use crate::config::{Config, MAX_METADATA_SIZE, MAX_SEARCH_RESULTS, MAX_TEXT_LENGTH};
use crate::database::{DatabaseClient, Memory, MemoryEvent};
use crate::embeddings::get_embedding_provider;
use crate::prompts::format_summarize_instructions;
use crate::types::*;
use crate::vector_store::{VectorSearchResult, VectorStore};
use anyhow::Result;
use chrono::Utc;
use dialoguer::{Confirm, Input, Select};
use directories::ProjectDirs;
use serde_json::{json, Value};
use sqlx::Row;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

// =============================================================================
// MCP Response Helpers
// =============================================================================

/// Build an MCP tool response with text content.
fn mcp_text_response<T: serde::Serialize>(payload: &T) -> Result<Value> {
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(payload)?
        }]
    }))
}

/// Build an MCP tool response with pretty-printed text content.
fn mcp_text_response_pretty<T: serde::Serialize>(payload: &T) -> Result<Value> {
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(payload)?
        }]
    }))
}

// =============================================================================
// Configuration
// =============================================================================


fn config_path() -> Result<PathBuf> {
    let proj = ProjectDirs::from("com", "memento", "memento")
        .ok_or_else(|| anyhow::anyhow!("Cannot resolve config dir"))?;
    Ok(proj.config_dir().join("config.json"))
}

fn save_config(db_type: &str, db_url: &str, provider: &str, model: &str, embedding_dim: Option<usize>) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::json!({
        "database_type": db_type,
        "database_url": db_url,
        "embedding_provider": provider,
        "embedding_model": model
    });
    if let Some(dim) = embedding_dim {
        json["embedding_dim"] = serde_json::json!(dim);
    }
    fs::write(path, serde_json::to_vec_pretty(&json)?)?;
    eprintln!("⚠️  Note: API keys are not stored in config. Use environment variables (MEMENTO_OPENAI_API_KEY or OPENAI_API_KEY) for sensitive credentials.");
    Ok(())
}

fn load_config_from_file() -> Option<serde_json::Value> {
    let path = config_path().ok()?;
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub fn run_init() -> Result<()> {
    eprintln!("\n🧠 Welcome to Memento - Universal Agent Memory Engine\n");
    let (db_type, db_url) = prompt_database_config()?;
    let (provider, model, embedding_dim) = prompt_embedding_config()?;
    save_config(&db_type, &db_url, &provider, &model, embedding_dim)?;
    eprintln!("\n✅ Configuration saved!");
    Ok(())
}

/// Check if configuration exists
pub fn config_exists() -> bool {
    config_path().map(|p| p.exists()).unwrap_or(false)
}

/// Validate PostgreSQL URL format
fn is_valid_postgres_url(url: &str) -> bool {
    (url.starts_with("postgresql://") || url.starts_with("postgres://"))
        && url.contains('@')
        && url.len() > 15
}

/// Validate local database path - returns None if valid, Some(error) if invalid
fn validate_local_path(path: &str) -> Option<String> {
    // Check for empty path
    if path.is_empty() {
        return Some("Path cannot be empty.".to_string());
    }
    
    // Check for invalid characters (basic check)
    let invalid_chars = ['<', '>', ':', '"', '|', '?', '*'];
    for c in invalid_chars {
        if path.contains(c) {
            return Some(format!("Path contains invalid character: '{}'", c));
        }
    }
    
    // Check it's not just whitespace
    if path.chars().all(|c| c.is_whitespace()) {
        return Some("Path cannot be only whitespace.".to_string());
    }
    
    // Check for obviously invalid patterns
    if path.contains("//") && !path.starts_with("//") {
        return Some("Path contains invalid double slashes.".to_string());
    }
    
    // Expand ~ to home directory and check if parent could exist
    let expanded = if path.starts_with("~/") {
        if let Some(base_dirs) = directories::BaseDirs::new() {
            base_dirs.home_dir().join(&path[2..])
        } else {
            return Some("Could not expand ~ to home directory.".to_string());
        }
    } else {
        std::path::PathBuf::from(path)
    };
    
    // Check if path is absolute and parent exists, or if it's relative (which is fine)
    if expanded.is_absolute() {
        // For absolute paths, check if parent directory exists or can be created
        if let Some(parent) = expanded.parent() {
            if !parent.exists() && parent.parent().map(|p| !p.exists()).unwrap_or(false) {
                return Some(format!("Parent directory does not exist: {}", parent.display()));
            }
        }
    }
    
    None // Valid
}

fn prompt_database_config() -> Result<(String, String)> {
    let storage_options = vec![
        "📁 Local Storage (SQLite - recommended for personal use)",
        "☁️  Cloud Database (PostgreSQL - for teams/production)",
    ];
    
    let storage_idx = Select::new()
        .with_prompt("Where would you like to store your memories?")
        .items(&storage_options)
        .default(0)
        .interact()?;
    
    let db_type = if storage_idx == 0 { "sqlite" } else { "postgresql" }.to_string();
    
    let db_url = if db_type == "postgresql" {
        // Cloud database - require valid URL
        eprintln!("\n📝 Enter your PostgreSQL connection URL");
        eprintln!("   Format: postgresql://user:password@host:port/database\n");
        
        loop {
            let url = Input::<String>::new()
                .with_prompt("PostgreSQL URL")
                .interact_text()?;
            
            if is_valid_postgres_url(&url) {
                break url;
            } else {
                eprintln!("❌ Invalid URL format. Please enter a valid PostgreSQL URL.");
                eprintln!("   Example: postgresql://myuser:mypassword@localhost:5432/memento\n");
            }
        }
    } else {
        // Local storage - ask for folder or use default
        let default_path = "./db/memento.db";
        eprintln!("\n📁 Choose where to store your local database");
        eprintln!("   Default: {}", default_path);
        eprintln!("   Press Enter to use the default, or enter a folder path.\n");
        
        loop {
            let folder = Input::<String>::new()
                .with_prompt("Database folder")
                .allow_empty(true)
                .interact_text()?;
            
            let folder = folder.trim();
            
            // Use default if empty
            if folder.is_empty() {
                break default_path.to_string();
            }
            
            // Validate the path
            if let Some(validation_error) = validate_local_path(folder) {
                eprintln!("❌ {}", validation_error);
                eprintln!("   Please enter a valid folder path (e.g., ./data, /home/user/memento, ~/memento)\n");
                continue;
            }
            
            // Normalize and construct full path
            let folder = folder.trim_end_matches('/');
            break format!("{}/memento.db", folder);
        }
    };
    
    Ok((db_type, db_url))
}

fn prompt_embedding_config() -> Result<(String, String, Option<usize>)> {
    let providers = vec![
        "🖥️  Local (free, runs on device, no API key needed)",
        "🌐 OpenAI (requires API key, better quality)",
    ];
    let provider_idx = Select::new()
        .with_prompt("Select embedding provider")
        .items(&providers)
        .default(0)
        .interact()?;
    
    let provider = if provider_idx == 0 { "local" } else { "openai" }.to_string();
    
    // If OpenAI selected, validate API key is set
    if provider == "openai" {
        let has_api_key = std::env::var("MEMENTO_OPENAI_API_KEY").is_ok();
        
        if !has_api_key {
            eprintln!("\n⚠️  MEMENTO_OPENAI_API_KEY not found in environment.");
            eprintln!("\n   To use OpenAI embeddings, set this environment variable:");
            eprintln!("   export MEMENTO_OPENAI_API_KEY=sk-...");
            eprintln!("\n   Add this to your shell profile (~/.bashrc, ~/.zshrc) or");
            eprintln!("   in your MCP client's environment configuration.\n");
            
            let proceed = Confirm::new()
                .with_prompt("Continue with OpenAI anyway? (You'll need to set the API key before use)")
                .default(false)
                .interact()?;
            
            if !proceed {
                eprintln!("\n📝 Switching to local embeddings instead.\n");
                return prompt_embedding_config_details("local");
            }
        } else {
            eprintln!("\n✅ OpenAI API key found in environment.\n");
        }
    }
    
    prompt_embedding_config_details(&provider)
}

fn prompt_embedding_config_details(provider: &str) -> Result<(String, String, Option<usize>)> {
    let default_model = if provider == "openai" {
        "text-embedding-3-small"
    } else {
        "Xenova/all-MiniLM-L6-v2"
    };
    
    let model = Input::<String>::new()
        .with_prompt("Embedding model")
        .with_initial_text(default_model)
        .interact_text()?;
    
    let default_dim = if provider == "openai" { 1536 } else { 384 };
    let dim_str = Input::<String>::new()
        .with_prompt("Embedding dimension (press Enter for default)")
        .with_initial_text(default_dim.to_string())
        .allow_empty(true)
        .interact_text()?;
    
    let embedding_dim = if dim_str.is_empty() {
        None
    } else {
        dim_str.parse().ok()
    };
    
    Ok((provider.to_string(), model, embedding_dim))
}

pub async fn start_mcp_server(
    database_type: Option<String>,
    database_url: Option<String>,
    embedding_dim_override: Option<usize>,
) -> Result<()> {
    // Check if this is first use (no config, no CLI args, no env vars)
    let has_cli_args = database_type.is_some() || database_url.is_some();
    let has_env_vars = std::env::var("MEMENTO_DATABASE_URL").is_ok();
    
    if !config_exists() && !has_cli_args && !has_env_vars {
        // First use - run interactive setup
        if atty::is(atty::Stream::Stdin) {
            run_init()?;
        } else {
            // Non-interactive mode (e.g., MCP client) - auto-create default config
            eprintln!("📝 First run detected. Creating default configuration...");
            save_config(
                "sqlite",
                "./db/memento.db",
                "local",
                "Xenova/all-MiniLM-L6-v2",
                Some(384),
            )?;
            eprintln!("✅ Default config created: SQLite + local embeddings");
            eprintln!("   Use memento.configure tool or 'memento init' to customize.\n");
        }
    }
    
    let file_cfg = load_config_from_file();
    let file_db_type = file_cfg.as_ref()
        .and_then(|v| v.get("database_type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let file_db_url = file_cfg.as_ref()
        .and_then(|v| v.get("database_url"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let file_provider = file_cfg.as_ref()
        .and_then(|v| v.get("embedding_provider"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let file_model = file_cfg.as_ref()
        .and_then(|v| v.get("embedding_model"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    
    let file_dim = file_cfg.as_ref()
        .and_then(|v| v.get("embedding_dim"))
        .and_then(|v| v.as_u64())
        .map(|d| d as usize);

    let (db_type, db_url) = if database_type.is_some() || database_url.is_some() {
        let db_type = database_type
            .or_else(|| std::env::var("MEMENTO_DATABASE_TYPE").ok())
            .unwrap_or_else(|| "sqlite".to_string());
        
        let db_url = database_url
            .or_else(|| std::env::var("MEMENTO_DATABASE_URL").ok())
            .or_else(|| {
                match db_type.as_str() {
                    "postgresql" | "postgres" => None,
                    _ => Some("./db/memento.db".to_string())
                }
            });
        
        let db_url = match db_type.as_str() {
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
                let path = db_url.unwrap_or_else(|| "./db/memento.db".to_string());
                let path = path.strip_prefix("sqlite://").unwrap_or(&path);
                format!("sqlite://{}", path)
            }
        };
        
        (db_type, db_url)
    } else {
        let db_type = std::env::var("MEMENTO_DATABASE_TYPE")
            .ok()
            .or_else(|| file_db_type)
            .unwrap_or_else(|| "sqlite".to_string());
        
        let db_url = std::env::var("MEMENTO_DATABASE_URL")
            .ok()
            .or_else(|| file_db_url)
            .or_else(|| {
                match db_type.as_str() {
                    "postgresql" | "postgres" => None,
                    _ => Some("./db/memento.db".to_string())
                }
            });
        
        if let Some(url) = db_url {
            let db_url = match db_type.as_str() {
                "postgresql" | "postgres" => {
                    if !url.starts_with("postgresql://") && !url.starts_with("postgres://") {
                        format!("postgresql://{}", url)
                    } else {
                        url
                    }
                }
                _ => {
                    let path = url.strip_prefix("sqlite://").unwrap_or(&url);
                    format!("sqlite://{}", path)
                }
            };
            (db_type, db_url)
        } else {
            let config_path_str = config_path().ok().map(|p| format!("{:?}", p));
            return Err(anyhow::anyhow!(
                "Memento MCP server is not configured. Run `memento init` in a terminal. Config path: {:?}",
                config_path_str
            ));
        }
    };

    eprintln!("📦 Using database: {} ({})", db_url, db_type);

    let provider = std::env::var("MEMENTO_EMBEDDING_PROVIDER")
        .ok()
        .or_else(|| file_provider)
        .unwrap_or_else(|| "local".to_string());
    
    let api_key = if provider == "openai" {
        std::env::var("MEMENTO_OPENAI_API_KEY").ok()
    } else {
        None
    };
    
    if provider == "openai" && api_key.is_none() {
        let config_path_str = config_path().ok().map(|p| format!("{:?}", p));
        return Err(anyhow::anyhow!(
            "OpenAI embedding provider requires API key. Set MEMENTO_OPENAI_API_KEY environment variable. Config path: {:?}",
            config_path_str
        ));
    }
    
    let model = std::env::var("MEMENTO_EMBEDDING_MODEL")
        .ok()
        .or_else(|| file_model)
        .unwrap_or_else(|| {
            let config = Config::from_env();
            config.embedding_model
        });
    
    let embedding_dim = embedding_dim_override
        .or_else(|| {
            std::env::var("MEMENTO_EMBEDDING_DIM")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .or(file_dim)
        .or_else(|| {
            let config = Config::from_env();
            config.embedding_dim
        });
    
    let embedding_provider = get_embedding_provider(
        provider.as_str(),
        api_key,
        Some(model),
        embedding_dim,
    )?;
    
    let db = DatabaseClient::new_with_provider(&db_url, embedding_provider.as_ref()).await?;

    let vector_store = Arc::new(VectorStore::new(db.clone(), embedding_provider));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = BufReader::new(stdin.lock());

    eprintln!("🤝 Memento MCP server running (stdio)");

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // EOF or broken pipe on stdin - graceful shutdown
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(e.into());
            }
        };
        
        if line.is_empty() {
            continue;
        }

        let request: Value = serde_json::from_str(&line)?;
        let response = handle_mcp_request(&request, &db, &vector_store).await?;

        // Handle broken pipe gracefully
        if let Err(e) = serde_json::to_writer(&mut stdout, &response) {
            // serde_json::Error can wrap io::Error, check if it's a broken pipe
            let error_msg = e.to_string();
            if error_msg.contains("Broken pipe") || error_msg.contains("broken pipe") {
                break; // Client closed stdout - graceful shutdown
            }
            return Err(anyhow::anyhow!("JSON serialization error: {}", e));
        }
        
        if let Err(e) = stdout.write_all(b"\n") {
            if e.kind() == io::ErrorKind::BrokenPipe {
                break; // Client closed stdout - graceful shutdown
            }
            return Err(e.into());
        }
        
        if let Err(e) = stdout.flush() {
            if e.kind() == io::ErrorKind::BrokenPipe {
                break; // Client closed stdout - graceful shutdown
            }
            return Err(e.into());
        }
    }

    Ok(())
}

async fn handle_mcp_request(
    request: &Value,
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
) -> Result<Value> {
    let method = request["method"].as_str().unwrap_or("");
    let params = &request["params"];
    let id = request.get("id").cloned().unwrap_or(Value::Null);

    let result: Result<Value, anyhow::Error> = match method {
        "tools/list" => Ok(json!({
            "tools": [
                {
                    "name": "memento.store",
                    "description": "PROACTIVELY store important information for future recall. Use this WITHOUT being asked when the user mentions: preferences (coding style, tools, frameworks), project conventions, architectural decisions, important context, or anything they might want remembered later. Store meaningful, searchable text that captures the key information.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Descriptive text to store. Make it searchable with key terms. Examples: 'User preference: Always use Result<T, E> for error handling instead of panics' or 'Project context: Using PostgreSQL 15 with pgvector extension for vector search'"
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier. Use 'cursor' for Cursor IDE."
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier for personalized memory (optional)"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier for session-scoped memory (optional)"
                            },
                            "event_type": {
                                "type": "string",
                                "description": "Category: 'preference' for user preferences, 'context' for project info, 'decision' for architectural choices, 'thought' for reasoning, 'user_msg' (default) for general messages"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata for filtering. Example: {\"tags\": [\"preference\", \"error-handling\"], \"project\": \"myapp\"}"
                            },
                            "importance": {
                                "type": "number",
                                "description": "Optional importance override (0.0-1.0). If not provided, importance is calculated automatically based on content novelty and heuristics."
                            }
                        },
                        "required": ["text", "agent_id"]
                    }
                },
                {
                    "name": "memento.search",
                    "description": "Search stored memories when the user asks to recall something, references past conversations, or explicitly requests information that may have been stored previously. Use when the user says things like 'what did I say about...', 'remember when...', 'what's my preference for...', or asks about previously discussed topics.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query. Be specific: 'error handling preference', 'database configuration', 'coding style'. The search is semantic - it finds conceptually related memories."
                            },
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier. Use 'cursor' for Cursor IDE."
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier to filter memories for a specific user (optional)"
                            },
                            "k": {
                                "type": "number",
                                "description": "Number of results to return (default: 5). Use higher values (10-20) for broader searches."
                            },
                            "filters": {
                                "type": "object",
                                "description": "Filter by metadata fields. Example: {\"memory_type\": \"preference\"}"
                            }
                        },
                        "required": ["query", "agent_id"]
                    }
                },
                {
                    "name": "memento.summarize",
                    "description": "Get unsummarized events that need to be consolidated into long-term memories. Returns events with instructions - YOU (the LLM) should analyze them, then call memento.store with a summary, and finally call memento.mark_summarized with the event IDs.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier. Use 'cursor' for Cursor IDE."
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier (optional)"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier to summarize a specific session (optional)"
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of events to return (default: 50)"
                            }
                        },
                        "required": ["agent_id"]
                    }
                },
                {
                    "name": "memento.mark_summarized",
                    "description": "Mark events as summarized after you have processed them and stored a summary via memento.store. Call this after successfully storing a summary to prevent re-processing the same events.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier. Use 'cursor' for Cursor IDE."
                            },
                            "event_ids": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Array of event IDs to mark as summarized"
                            }
                        },
                        "required": ["agent_id", "event_ids"]
                    }
                },
                {
                    "name": "memento.forget",
                    "description": "Remove memories when user explicitly requests it ('forget that', 'delete my preference about...', 'I changed my mind about...') or when information becomes outdated. Provide either a specific memory_id OR a query to find and remove matching memories.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Agent identifier. Use 'cursor' for Cursor IDE."
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier (optional)"
                            },
                            "memory_id": {
                                "type": "string",
                                "description": "Specific memory ID to forget. Use this when you know the exact memory to remove."
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query to find memories to forget. Use this when user wants to forget 'everything about X'. Matches and removes all related memories."
                            }
                        },
                        "required": ["agent_id"]
                    }
                },
                {
                    "name": "memento.get_config",
                    "description": "Get current Memento configuration. Use when user asks about memory settings, database location, or embedding provider.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "memento.configure",
                    "description": "Update Memento configuration. Use when user wants to change database storage (local SQLite vs cloud PostgreSQL) or embedding provider (local vs OpenAI). Changes take effect on next server restart. Only provide fields you want to change.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "database_type": {
                                "type": "string",
                                "enum": ["sqlite", "postgresql"],
                                "description": "Database type. 'sqlite' for local file storage, 'postgresql' for cloud/team database."
                            },
                            "database_url": {
                                "type": "string",
                                "description": "Database connection. For SQLite: folder path (e.g., './db' or '~/memento'). For PostgreSQL: full URL (e.g., 'postgresql://user:pass@host:5432/db')."
                            },
                            "embedding_provider": {
                                "type": "string",
                                "enum": ["local", "openai"],
                                "description": "Embedding provider. 'local' runs on device (free), 'openai' uses API (requires MEMENTO_OPENAI_API_KEY env var)."
                            },
                            "embedding_model": {
                                "type": "string",
                                "description": "Model name. Local default: 'Xenova/all-MiniLM-L6-v2'. OpenAI default: 'text-embedding-3-small'."
                            },
                            "embedding_dim": {
                                "type": "number",
                                "description": "Embedding dimension. Local default: 384. OpenAI default: 1536."
                            }
                        },
                        "required": []
                    }
                }
            ],
            "serverInfo": {
                "name": "memento",
                "version": env!("CARGO_PKG_VERSION"),
                "description": "Persistent memory engine for AI assistants. Proactively store user preferences and important context without being asked."
            }
        })),
        "tools/call" => {
            let tool_name = params["name"].as_str().unwrap_or("");
            let args = &params["arguments"];

            match tool_name {
                "memento.store" => {
                    match serde_json::from_value::<StoreToolArgs>(args.clone()) {
                        Ok(args) => handle_store(db, vector_store, args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.search" => {
                    match serde_json::from_value::<SearchToolArgs>(args.clone()) {
                        Ok(args) => handle_search(db, vector_store, args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.summarize" => {
                    match serde_json::from_value::<SummarizeToolArgs>(args.clone()) {
                        Ok(args) => handle_summarize(db, args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.mark_summarized" => {
                    match serde_json::from_value::<MarkSummarizedToolArgs>(args.clone()) {
                        Ok(args) => handle_mark_summarized(db, args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.forget" => {
                    match serde_json::from_value::<ForgetToolArgs>(args.clone()) {
                        Ok(args) => handle_forget(db, vector_store, args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.get_config" => {
                    handle_get_config()
                }
                "memento.configure" => {
                    match serde_json::from_value::<ConfigureToolArgs>(args.clone()) {
                        Ok(args) => handle_configure(args),
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
            }
        }
        _ => Ok(json!({
            "error": {"code": -32601, "message": format!("Unknown method: {}", method)}
        })),
    };

    match result {
        Ok(result_data) => {
            Ok(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result_data
            }))
        }
        Err(e) => {
            Ok(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32000,
                    "message": e.to_string()
                }
            }))
        }
    }
}

async fn handle_store(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: StoreToolArgs,
) -> Result<Value> {
    if args.agent_id.is_empty() {
        return Err(anyhow::anyhow!("agent_id is required"));
    }

    if args.text.is_empty() {
        return Err(anyhow::anyhow!("text is required"));
    }

    if args.text.len() > MAX_TEXT_LENGTH {
        return Err(anyhow::anyhow!(
            "text exceeds maximum length of {} bytes ({} provided)",
            MAX_TEXT_LENGTH,
            args.text.len()
        ));
    }

    // Validate and clamp importance override to [0.0, 1.0]
    let importance_override = args.importance.map(|imp| imp.clamp(0.0, 1.0));

    // Validate metadata size if provided
    if let Some(ref metadata) = args.metadata {
        let metadata_json = serde_json::to_string(metadata).unwrap_or_default();
        if metadata_json.len() > MAX_METADATA_SIZE {
            return Err(anyhow::anyhow!(
                "metadata exceeds maximum size of {} bytes ({} provided)",
                MAX_METADATA_SIZE,
                metadata_json.len()
            ));
        }
    }

    let event_type = args.event_type.unwrap_or_else(|| "user_msg".to_string());

    let event_id = Uuid::new_v4().to_string();
    let event = MemoryEvent {
        id: event_id.clone(),
        agent_id: args.agent_id.clone(),
        user_id: args.user_id.clone(),
        session_id: args.session_id.clone(),
        event_type: event_type.clone(),
        content: args.text.clone(),
        metadata: args
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default()),
        created_at: Utc::now(),
        summarized_at: None,
    };

    db.insert_event(&event).await?;

    let mut memory_id: Option<String> = None;
    let word_count = args.text.split_whitespace().count();
    let should_promote = word_count >= 6
        && args.text.len() >= 40
        && matches!(event_type.as_str(), "user_msg" | "thought")
        && !args.text.trim().ends_with('?'); // Questions are often ephemeral
    
    if should_promote {
        // Compute embedding first - needed for both importance scoring and storage
        let embedding = match vector_store.compute_embedding(&args.text).await {
            Ok(emb) => emb,
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to compute embedding: {}", e));
            }
        };

        // Calculate importance using multiple signals (content heuristics + novelty)
        // Uses the pre-validated and clamped importance_override
        let importance = crate::scoring::calculate_importance(
            &args.text,
            &embedding,
            vector_store,
            &args.agent_id,
            args.user_id.as_deref(),
            importance_override,
        )
        .await;

        let memory = Memory {
            id: Uuid::new_v4().to_string(),
            agent_id: args.agent_id.clone(),
            user_id: args.user_id.clone(),
            session_id: args.session_id.clone(),
            memory_type: "episodic".to_string(),
            text: args.text.clone(),
            importance,
            is_active: true,
            supersedes_id: None,
            source_event_ids: Some(json!([event_id]).to_string()),
            metadata: args
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
            last_accessed_at: None,
            created_at: Utc::now(),
        };

        let mem_id = memory.id.clone();
        db.insert_memory(&memory).await?;

        let mut vector_metadata = Metadata::new();
        vector_metadata.insert("agent_id".to_string(), json!(args.agent_id));
        if let Some(user_id) = &args.user_id {
            vector_metadata.insert("user_id".to_string(), json!(user_id));
        }
        if let Some(session_id) = &args.session_id {
            vector_metadata.insert("session_id".to_string(), json!(session_id));
        }
        vector_metadata.insert("memory_type".to_string(), json!("episodic"));
        if let Some(metadata) = &args.metadata {
            for (k, v) in metadata {
                vector_metadata.insert(k.clone(), v.clone());
            }
        }

        // Store embedding (already computed above for importance scoring)
        // If embedding add fails, deactivate the memory to avoid inconsistent state
        if let Err(e) = vector_store.add(&mem_id, &args.text, Some(embedding), vector_metadata).await {
            // Cleanup: deactivate memory if embedding storage fails
            let _ = db.soft_delete_memory(&mem_id).await;
            return Err(anyhow::anyhow!("Failed to store embedding: {}", e));
        }

        // Insert provenance links to track which events created this memory
        db.insert_memory_sources(&mem_id, &[event_id.clone()]).await?;

        memory_id = Some(mem_id);
    }

    mcp_text_response(&json!({
        "ok": true,
        "event_id": event_id,
        "memory_id": memory_id
    }))
}
async fn handle_search(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: SearchToolArgs,
) -> Result<Value> {
    if args.agent_id.is_empty() || args.query.is_empty() {
        return Err(anyhow::anyhow!("agent_id and query are required"));
    }

    // Clamp k to prevent unbounded resource usage
    let k = args.k.unwrap_or(5).min(MAX_SEARCH_RESULTS);
    let filters = args.filters.unwrap_or_default();

    // Try vector search first (semantic search)
    let mut vector_results = vector_store
        .search(
            &args.query,
            k,
            filters,
            Some(&args.agent_id),
            args.user_id.as_deref(),
        )
        .await?;

    if vector_results.is_empty() {
        vector_results = keyword_search_fallback(
            db,
            &args.query,
            k,
            Some(&args.agent_id),
            args.user_id.as_deref(),
        )
        .await?;
    }

    let memory_ids: Vec<String> = vector_results.iter().map(|r| r.memory_id.clone()).collect();
    let memories = db.get_memories_by_ids(&memory_ids).await?;
    
    // Boost importance for accessed memories (also updates last_accessed_at)
    for memory_id in &memory_ids {
        if let Some(memory) = memories.get(memory_id) {
            if memory.is_active {
                let _ = db.boost_importance(
                    &memory.id,
                    crate::scoring::ACCESS_BOOST,
                    crate::scoring::MAX_IMPORTANCE,
                ).await;
            }
        }
    }

    let mut results = Vec::new();
    for result in vector_results {
        if let Some(memory) = memories.get(&result.memory_id) {
            if memory.is_active {
                let mut metadata = if let Some(meta_str) = &memory.metadata {
                    serde_json::from_str(meta_str).unwrap_or_default()
                } else {
                    Metadata::new()
                };

                for (k, v) in result.metadata {
                    metadata.insert(k, v);
                }

                results.push(json!({
                    "memory_id": memory.id,
                    "text": memory.text,
                    "score": result.score,
                    "metadata": metadata
                }));
            }
        }
    }

    mcp_text_response(&json!({
        "ok": true,
        "results": results
    }))
}

/// Escapes SQL LIKE wildcard characters to prevent unintended pattern matching.
/// Escapes: % (any chars), _ (single char), \ (escape char)
#[doc(hidden)]
pub fn escape_like_pattern(query: &str) -> String {
    query
        .replace('\\', "\\\\") // Escape backslash first
        .replace('%', "\\%")   // Escape percent
        .replace('_', "\\_")   // Escape underscore
}

/// Keyword search fallback when vector search returns no results.
/// Uses SQL LIKE for simple text matching.
/// 
/// # Security
/// Query is escaped to prevent SQL LIKE wildcard injection.
async fn keyword_search_fallback(
    db: &DatabaseClient,
    query: &str,
    k: usize,
    agent_id: Option<&str>,
    user_id: Option<&str>,
) -> Result<Vec<VectorSearchResult>> {
    // Escape LIKE wildcards to prevent injection (e.g., "100%" matching everything)
    let escaped_query = escape_like_pattern(query);
    let search_pattern = format!("%{}%", escaped_query);
    let mut results = Vec::new();

    match db {
        DatabaseClient::Sqlite(pool) => {
            let mut sql = String::from(
                "SELECT id, text, metadata FROM memories WHERE is_active = 1 AND text LIKE ?1 ESCAPE '\\'"
            );
            let mut binds: Vec<String> = vec![search_pattern.clone()];
            let mut bind_idx = 2;

            if let Some(agent_id) = agent_id {
                sql.push_str(&format!(" AND agent_id = ?{}", bind_idx));
                binds.push(agent_id.to_string());
                bind_idx += 1;
            }

            if let Some(user_id) = user_id {
                sql.push_str(&format!(" AND user_id = ?{}", bind_idx));
                binds.push(user_id.to_string());
            }

            let mut q = sqlx::query(&sql);
            for b in &binds {
                q = q.bind(b);
            }

            let rows = q.fetch_all(pool).await?;
            
            for row in rows.into_iter().take(k) {
                let memory_id: String = row.try_get(&"id")?;
                let text: String = row.try_get(&"text")?;
                let metadata_str: Option<String> = row.try_get(&"metadata")?;
                
                // Simple relevance score based on query term frequency
                let score = text.to_lowercase().matches(&query.to_lowercase()).count() as f64 / 10.0;
                
                let metadata: Metadata = if let Some(meta_str) = metadata_str {
                    serde_json::from_str(&meta_str).unwrap_or_default()
                } else {
                    Metadata::new()
                };

                results.push(VectorSearchResult {
                    memory_id,
                    score: score.min(1.0), // Cap at 1.0
                    metadata,
                });
            }
        }
        DatabaseClient::Postgres(pool) => {
            let mut sql = String::from(
                "SELECT id, text, metadata FROM memories WHERE is_active = TRUE AND text ILIKE $1 ESCAPE '\\'"
            );
            let mut binds: Vec<String> = vec![search_pattern.clone()];
            let mut bind_idx = 2;

            if let Some(agent_id) = agent_id {
                sql.push_str(&format!(" AND agent_id = ${}", bind_idx));
                binds.push(agent_id.to_string());
                bind_idx += 1;
            }

            if let Some(user_id) = user_id {
                sql.push_str(&format!(" AND user_id = ${}", bind_idx));
                binds.push(user_id.to_string());
            }

            let mut q = sqlx::query(&sql);
            for b in &binds {
                q = q.bind(b);
            }

            let rows = q.fetch_all(pool).await?;
            
            for row in rows.into_iter().take(k) {
                let memory_id: String = row.try_get(&"id")?;
                let text: String = row.try_get(&"text")?;
                let metadata_str: Option<String> = row.try_get(&"metadata")?;
                
                // Simple relevance score based on query term frequency
                let score = text.to_lowercase().matches(&query.to_lowercase()).count() as f64 / 10.0;
                
                let metadata: Metadata = if let Some(meta_str) = metadata_str {
                    serde_json::from_str(&meta_str).unwrap_or_default()
                } else {
                    Metadata::new()
                };

                results.push(VectorSearchResult {
                    memory_id,
                    score: score.min(1.0), // Cap at 1.0
                    metadata,
                });
            }
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(results)
}

async fn handle_summarize(
    db: &DatabaseClient,
    args: SummarizeToolArgs,
) -> Result<Value> {
    if args.agent_id.is_empty() {
        return Err(anyhow::anyhow!("agent_id is required"));
    }

    let limit = args.limit.unwrap_or(50);
    
    // Get unsummarized events
    let events = db.list_unsummarized_events(
        &args.agent_id,
        args.user_id.as_deref(),
        args.session_id.as_deref(),
        limit,
    ).await?;

    if events.is_empty() {
        return mcp_text_response(&json!({
            "ok": true,
            "has_events": false,
            "message": "No unsummarized events found. Nothing to summarize."
        }));
    }

    // Format events for the LLM
    let event_ids: Vec<String> = events.iter().map(|e| e.id.clone()).collect();
    let events_text: Vec<String> = events.iter().map(|e| {
        format!(
            "[{}] {}: {}",
            e.created_at.format("%Y-%m-%d %H:%M"),
            e.event_type,
            e.content
        )
    }).collect();

    let instructions = format_summarize_instructions(
        events.len(),
        &events_text.join("\n"),
        &args.agent_id,
        &event_ids,
    );

    mcp_text_response_pretty(&json!({
        "ok": true,
        "has_events": true,
        "event_count": events.len(),
        "event_ids": event_ids,
        "instructions": instructions
    }))
}

async fn handle_mark_summarized(
    db: &DatabaseClient,
    args: MarkSummarizedToolArgs,
) -> Result<Value> {
    if args.agent_id.is_empty() {
        return Err(anyhow::anyhow!("agent_id is required"));
    }

    if args.event_ids.is_empty() {
        return Err(anyhow::anyhow!("event_ids cannot be empty"));
    }

    db.mark_events_summarized(&args.agent_id, &args.event_ids).await?;

    mcp_text_response(&json!({
        "ok": true,
        "marked": args.event_ids.len(),
        "message": format!("Marked {} event(s) as summarized", args.event_ids.len())
    }))
}

async fn handle_forget(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: ForgetToolArgs,
) -> Result<Value> {
    if args.agent_id.is_empty() {
        return Err(anyhow::anyhow!("agent_id is required"));
    }

    let mut deleted = 0;

    if let Some(memory_id) = args.memory_id {
        if let Some(memory) = db.get_memory(&memory_id).await? {
            if memory.agent_id == args.agent_id {
                db.soft_delete_memory(&memory_id).await?;
                vector_store.delete(&memory_id).await?;
                deleted = 1;
            }
        }
    } else if let Some(query) = args.query {
        let vector_results = vector_store
            .search(
                &query,
                20,
                Metadata::new(),
                Some(&args.agent_id),
                args.user_id.as_deref(),
            )
            .await?;

        for result in vector_results {
            if let Some(memory) = db.get_memory(&result.memory_id).await? {
                if memory.is_active {
                    db.soft_delete_memory(&memory.id).await?;
                    vector_store.delete(&memory.id).await?;
                    deleted += 1;
                }
            }
        }
    } else {
        return Err(anyhow::anyhow!("Either memory_id or query is required"));
    }

    mcp_text_response(&json!({
        "ok": true,
        "deleted": deleted
    }))
}

fn handle_get_config() -> Result<Value> {
    let config = load_config_from_file();
    
    match config {
        Some(cfg) => {
            let db_type = cfg.get("database_type")
                .and_then(|v| v.as_str())
                .unwrap_or("sqlite");
            let db_url = cfg.get("database_url")
                .and_then(|v| v.as_str())
                .unwrap_or("./db/memento.db");
            let provider = cfg.get("embedding_provider")
                .and_then(|v| v.as_str())
                .unwrap_or("local");
            let model = cfg.get("embedding_model")
                .and_then(|v| v.as_str())
                .unwrap_or("Xenova/all-MiniLM-L6-v2");
            let dim = cfg.get("embedding_dim")
                .and_then(|v| v.as_u64())
                .map(|d| d as usize);
            
            let config_path_str = config_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string());
            
            mcp_text_response(&json!({
                "database_type": db_type,
                "database_url": db_url,
                "embedding_provider": provider,
                "embedding_model": model,
                "embedding_dim": dim,
                "config_path": config_path_str,
                "note": "Changes via memento.configure take effect after restarting the MCP server."
            }))
        }
        None => {
            mcp_text_response(&json!({
                "error": "No configuration file found",
                "note": "Memento is using default settings. Use memento.configure to customize."
            }))
        }
    }
}

fn handle_configure(args: ConfigureToolArgs) -> Result<Value> {
    use crate::types::{DatabaseType, EmbeddingProviderType};
    
    // Load existing config or start with defaults
    let mut config = load_config_from_file().unwrap_or_else(|| {
        json!({
            "database_type": "sqlite",
            "database_url": "./db/memento.db",
            "embedding_provider": "local",
            "embedding_model": "Xenova/all-MiniLM-L6-v2",
            "embedding_dim": 384
        })
    });
    
    let mut changes = Vec::new();
    
    // Apply database_type (already validated by serde deserialization)
    if let Some(db_type) = &args.database_type {
        config["database_type"] = json!(db_type.to_string());
        changes.push(format!("database_type: {}", db_type));
    }
    
    // Validate and apply database_url
    if let Some(db_url) = &args.database_url {
        let is_postgresql = args.database_type == Some(DatabaseType::Postgresql)
            || config["database_type"].as_str() == Some("postgresql");
        
        if is_postgresql {
            // Validate PostgreSQL URL
            if !is_valid_postgres_url(db_url) {
                return Err(anyhow::anyhow!(
                    "Invalid PostgreSQL URL. Format: postgresql://user:password@host:port/database"
                ));
            }
            config["database_url"] = json!(db_url);
        } else {
            // SQLite - treat as folder path, append memento.db
            let path = db_url.trim().trim_end_matches('/');
            let full_path = if path.ends_with(".db") {
                path.to_string()
            } else {
                format!("{}/memento.db", path)
            };
            config["database_url"] = json!(full_path);
        }
        changes.push(format!("database_url: {}", config["database_url"].as_str().unwrap()));
    }
    
    // Apply embedding_provider (already validated by serde deserialization)
    if let Some(provider) = &args.embedding_provider {
        if *provider == EmbeddingProviderType::Openai {
            // Check if API key is set
            if std::env::var("MEMENTO_OPENAI_API_KEY").is_err() {
                return Err(anyhow::anyhow!(
                    "OpenAI provider requires MEMENTO_OPENAI_API_KEY environment variable. \
                    Please set this in your shell profile or MCP client configuration before switching to OpenAI."
                ));
            }
        }
        config["embedding_provider"] = json!(provider.to_string());
        changes.push(format!("embedding_provider: {}", provider));
        
        // Auto-update model and dim defaults when switching providers
        if args.embedding_model.is_none() {
            let default_model = match provider {
                EmbeddingProviderType::Openai => "text-embedding-3-small",
                EmbeddingProviderType::Local => "Xenova/all-MiniLM-L6-v2",
            };
            config["embedding_model"] = json!(default_model);
            changes.push(format!("embedding_model: {} (auto)", default_model));
        }
        if args.embedding_dim.is_none() {
            let default_dim = match provider {
                EmbeddingProviderType::Openai => 1536,
                EmbeddingProviderType::Local => 384,
            };
            config["embedding_dim"] = json!(default_dim);
            changes.push(format!("embedding_dim: {} (auto)", default_dim));
        }
    }
    
    // Apply embedding_model
    if let Some(model) = &args.embedding_model {
        config["embedding_model"] = json!(model);
        changes.push(format!("embedding_model: {}", model));
    }
    
    // Apply embedding_dim
    if let Some(dim) = args.embedding_dim {
        config["embedding_dim"] = json!(dim);
        changes.push(format!("embedding_dim: {}", dim));
    }
    
    if changes.is_empty() {
        return mcp_text_response(&json!({
            "ok": false,
            "message": "No changes provided. Specify at least one field to update."
        }));
    }
    
    // Save the updated config
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(&config)?)?;
    
    mcp_text_response(&json!({
        "ok": true,
        "changes": changes,
        "message": "Configuration updated. Restart the MCP server for changes to take effect.",
        "config_path": path.display().to_string()
    }))
}


