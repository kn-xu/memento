use crate::config::Config;
use crate::database::{DatabaseClient, Memory, MemoryEvent};
use crate::embeddings::get_embedding_provider;
use crate::types::*;
use crate::vector_store::VectorStore;
use anyhow::Result;
use chrono::Utc;
use dialoguer::{Input, Select};
use directories::ProjectDirs;
use serde_json::json;
use serde_json::Value;
use sqlx::Row;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;


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
    let (db_type, db_url) = prompt_database_config()?;
    let (provider, model, embedding_dim) = prompt_embedding_config()?;
    save_config(&db_type, &db_url, &provider, &model, embedding_dim)?;
    eprintln!("✅ Configuration saved! You can now run: memento mcp");
    Ok(())
}

fn prompt_database_config() -> Result<(String, String)> {
    let db_types = vec!["sqlite", "postgresql"];
    let db_type_idx = Select::new()
        .with_prompt("Select database type")
        .items(&db_types)
        .default(0)
        .interact()?;
    
    let db_type = db_types[db_type_idx].to_string();
    
    let db_url = if db_type == "postgresql" {
        Input::<String>::new()
            .with_prompt("Enter PostgreSQL connection URL")
            .with_initial_text("postgresql://user:password@localhost:5432/memento")
            .interact_text()?
    } else {
        Input::<String>::new()
            .with_prompt("Enter SQLite database path")
            .with_initial_text("./memento.db")
            .interact_text()?
    };
    
    Ok((db_type, db_url))
}

fn prompt_embedding_config() -> Result<(String, String, Option<usize>)> {
    let providers = vec!["local", "openai"];
    let provider_idx = Select::new()
        .with_prompt("Select embedding provider")
        .items(&providers)
        .default(0)
        .interact()?;
    
    let provider = providers[provider_idx].to_string();
    
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
    
    if provider == "openai" {
        eprintln!("⚠️  Note: Set MEMENTO_OPENAI_API_KEY or OPENAI_API_KEY environment variable for OpenAI API access.");
    }
    
    Ok((provider, model, embedding_dim))
}

pub async fn start_mcp_server(
    database_type: Option<String>,
    database_url: Option<String>,
    embedding_dim_override: Option<usize>,
) -> Result<()> {
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
                    "postgresql" | "postgres" => {
                        std::env::var("DATABASE_URL").ok()
                    }
                    _ => Some("./memento.db".to_string())
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
                let path = db_url.unwrap_or_else(|| "./memento.db".to_string());
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
                    "postgresql" | "postgres" => {
                        std::env::var("DATABASE_URL").ok()
                    }
                    _ => Some("./memento.db".to_string())
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
        .or_else(|| std::env::var("EMBEDDING_PROVIDER").ok())
        .or_else(|| file_provider)
        .unwrap_or_else(|| "local".to_string());
    
    let api_key = if provider == "openai" {
        std::env::var("MEMENTO_OPENAI_API_KEY")
            .ok()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
    } else {
        None
    };
    
    if provider == "openai" && api_key.is_none() {
        let config_path_str = config_path().ok().map(|p| format!("{:?}", p));
        return Err(anyhow::anyhow!(
            "OpenAI embedding provider requires API key. Set MEMENTO_OPENAI_API_KEY or OPENAI_API_KEY environment variable. Config path: {:?}",
            config_path_str
        ));
    }
    
    let model = std::env::var("MEMENTO_EMBEDDING_MODEL")
        .ok()
        .or_else(|| std::env::var("EMBEDDING_MODEL").ok())
        .or_else(|| file_model)
        .unwrap_or_else(|| {
            let config = Config::from_env();
            config.embedding_model
        });
    
    let embedding_dim = embedding_dim_override
        .or_else(|| {
            std::env::var("MEMENTO_EMBEDDING_DIM")
                .ok()
                .or_else(|| std::env::var("EMBEDDING_DIM").ok())
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
                    "description": "Summarize recent events into durable memories. Use periodically to consolidate conversation history into long-term memory. (Coming soon - not yet implemented)",
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
                            }
                        },
                        "required": ["agent_id"]
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
                        Ok(args) => handle_summarize(args).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid arguments: {}", e)),
                    }
                }
                "memento.forget" => {
                    match serde_json::from_value::<ForgetToolArgs>(args.clone()) {
                        Ok(args) => handle_forget(db, vector_store, args).await,
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
        let memory = Memory {
            id: Uuid::new_v4().to_string(),
            agent_id: args.agent_id.clone(),
            user_id: args.user_id.clone(),
            session_id: args.session_id.clone(),
            memory_type: "episodic".to_string(),
            text: args.text.clone(),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: Some(json!([event_id]).to_string()),
            metadata: args
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
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

        // VectorStore::add() computes the embedding (since we pass None) and then
        // UPDATEs the memories.embedding column, so embeddings are properly stored
        // If embedding add fails, deactivate the memory to avoid inconsistent state
        if let Err(e) = vector_store.add(&mem_id, &args.text, None, vector_metadata).await {
            // Cleanup: deactivate memory if embedding storage fails
            let _ = db.soft_delete_memory(&mem_id).await;
            return Err(anyhow::anyhow!("Failed to store embedding: {}", e));
        }

        // TODO: Insert provenance links via db.insert_memory_sources(&mem_id, &[event_id]).await?
        // Currently not called - memory_sources table exists but is not populated

        memory_id = Some(mem_id);
    }

    let payload = json!({
        "ok": true,
        "event_id": event_id,
        "memory_id": memory_id
    });
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&payload)?
        }]
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

    let k = args.k.unwrap_or(5);
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
    
    for memory_id in &memory_ids {
        if let Some(memory) = memories.get(memory_id) {
            if memory.is_active {
                let _ = db.update_memory_access(&memory.id).await;
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

    let payload = json!({
        "ok": true,
        "results": results
    });
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&payload)?
        }]
    }))
}

/// Keyword search fallback when vector search returns no results
/// Uses SQL LIKE for simple text matching
async fn keyword_search_fallback(
    db: &DatabaseClient,
    query: &str,
    k: usize,
    agent_id: Option<&str>,
    user_id: Option<&str>,
) -> Result<Vec<crate::vector_store::VectorSearchResult>> {
    use crate::vector_store::VectorSearchResult;
    use crate::types::Metadata;
    
    let search_pattern = format!("%{}%", query);
    let mut results = Vec::new();

    match db {
        DatabaseClient::Sqlite(pool) => {
            let mut sql = String::from(
                "SELECT id, text, metadata FROM memories WHERE is_active = 1 AND text LIKE ?1 AND (expires_at IS NULL OR expires_at > datetime('now'))"
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
                "SELECT id, text, metadata FROM memories WHERE is_active = TRUE AND text ILIKE $1 AND (expires_at IS NULL OR expires_at > NOW())"
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

async fn handle_summarize(_args: SummarizeToolArgs) -> Result<Value> {
    // TODO: Implement LLM-based summarization
    // This should:
    let payload = json!({
        "ok": true,
        "created": 0,
        "updated": 0,
        "message": "Summarization coming soon"
    });
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&payload)?
        }]
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

    let payload = json!({
        "ok": true,
        "deleted": deleted
    });
    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string(&payload)?
        }]
    }))
}

