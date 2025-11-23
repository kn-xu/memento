use crate::config::Config;
use crate::database::{DatabaseClient, Memory, MemoryEvent};
use crate::embeddings::get_embedding_provider;
use crate::types::*;
use crate::vector_store::VectorStore;
use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use serde_json::Value;
use sqlx::Row;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use uuid::Uuid;

pub async fn start_mcp_server(
    database_type: Option<String>,
    database_url: Option<String>,
) -> Result<()> {
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
                anyhow::anyhow!(
                    "PostgreSQL requires database URL. Set MEMENTO_DATABASE_URL or DATABASE_URL environment variable, or use --database-url flag"
                )
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

    eprintln!("📦 Using database: {} ({})", db_url, db_type);

    let mut config = Config::from_env();
    config.database_url = db_url.clone();
    
    let db = DatabaseClient::new(&db_url).await?;

    let embedding_provider = get_embedding_provider(
        match config.embedding_provider {
            crate::config::EmbeddingProvider::OpenAi => "openai",
            crate::config::EmbeddingProvider::Local => "local",
        },
        config.openai_api_key.clone(),
        Some(config.embedding_model.clone()),
    )?;

    let vector_store = Arc::new(VectorStore::new(db.clone(), embedding_provider));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = BufReader::new(stdin.lock());

    eprintln!("🤝 Memento MCP server running (stdio)");

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let request: Value = serde_json::from_str(&line)?;
        let response = handle_mcp_request(&request, &db, &vector_store).await?;

        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
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
                    "description": "Store a memory event and optionally create a derived memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text content to store"},
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "user_id": {"type": "string", "description": "User identifier (optional)"},
                            "session_id": {"type": "string", "description": "Session identifier (optional)"},
                            "event_type": {"type": "string", "description": "Event type (default: user_msg)"},
                            "metadata": {"type": "object", "description": "Additional metadata (optional)"}
                        },
                        "required": ["text", "agent_id"]
                    }
                },
                {
                    "name": "memento.search",
                    "description": "Semantic search for memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "user_id": {"type": "string", "description": "User identifier (optional)"},
                            "k": {"type": "number", "description": "Number of results (default: 5)"},
                            "filters": {"type": "object", "description": "Additional filters (optional)"}
                        },
                        "required": ["query", "agent_id"]
                    }
                },
                {
                    "name": "memento.summarize",
                    "description": "Summarize recent events into durable memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "user_id": {"type": "string", "description": "User identifier (optional)"},
                            "session_id": {"type": "string", "description": "Session identifier (optional)"}
                        },
                        "required": ["agent_id"]
                    }
                },
                {
                    "name": "memento.forget",
                    "description": "Forget a memory by ID or query",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "user_id": {"type": "string", "description": "User identifier (optional)"},
                            "memory_id": {"type": "string", "description": "Memory ID to forget (optional)"},
                            "query": {"type": "string", "description": "Query to find memories to forget (optional)"}
                        },
                        "required": ["agent_id"]
                    }
                }
            ]
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

    // Wrap result in JSON-RPC response format
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

    // Use default event_type if not provided (same default used for event creation)
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
    };

    db.insert_event(&event).await?;

    let mut memory_id: Option<String> = None;
    // Create durable memory if text is substantial and event_type is one that should be stored
    if args.text.len() > 10
        && matches!(event_type.as_str(), "user_msg" | "thought")
    {
        let memory = Memory {
            id: Uuid::new_v4().to_string(),
            agent_id: args.agent_id.clone(),
            user_id: args.user_id.clone(),
            session_id: args.session_id.clone(),
            memory_type: "episodic".to_string(),
            text: args.text.clone(),
            embedding: None,
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
        vector_store
            .add(&mem_id, &args.text, None, vector_metadata)
            .await?;

        memory_id = Some(mem_id);
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": json!({
                "ok": true,
                "event_id": event_id,
                "memory_id": memory_id
            })
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

    // If vector search returns no results, fall back to keyword search
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

    let mut results = Vec::new();
    for result in vector_results {
        if let Some(memory) = db.get_memory(&result.memory_id).await? {
            if memory.is_active {
                db.update_memory_access(&memory.id).await?;

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

    Ok(json!({
        "content": [{
            "type": "text",
            "text": json!({
                "ok": true,
                "results": results
            })
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
            let mut sql_query = sqlx::query(
                "SELECT id, text, metadata FROM memories 
                 WHERE is_active = 1 AND text LIKE ?1"
            )
            .bind(&search_pattern);

            if let Some(agent_id) = agent_id {
                sql_query = sqlx::query(
                    "SELECT id, text, metadata FROM memories 
                     WHERE is_active = 1 AND text LIKE ?1 AND agent_id = ?2"
                )
                .bind(&search_pattern)
                .bind(agent_id);
            }

            if let Some(user_id) = user_id {
                if agent_id.is_some() {
                    sql_query = sqlx::query(
                        "SELECT id, text, metadata FROM memories 
                         WHERE is_active = 1 AND text LIKE ?1 AND agent_id = ?2 AND user_id = ?3"
                    )
                    .bind(&search_pattern)
                    .bind(agent_id.unwrap())
                    .bind(user_id);
                } else {
                    sql_query = sqlx::query(
                        "SELECT id, text, metadata FROM memories 
                         WHERE is_active = 1 AND text LIKE ?1 AND user_id = ?2"
                    )
                    .bind(&search_pattern)
                    .bind(user_id);
                }
            }

            let rows = sql_query.fetch_all(pool).await?;
            
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
            let mut sql_query = sqlx::query(
                "SELECT id, text, metadata FROM memories 
                 WHERE is_active = TRUE AND text ILIKE $1"
            )
            .bind(&search_pattern);

            if let Some(agent_id) = agent_id {
                sql_query = sqlx::query(
                    "SELECT id, text, metadata FROM memories 
                     WHERE is_active = TRUE AND text ILIKE $1 AND agent_id = $2"
                )
                .bind(&search_pattern)
                .bind(agent_id);
            }

            if let Some(user_id) = user_id {
                if agent_id.is_some() {
                    sql_query = sqlx::query(
                        "SELECT id, text, metadata FROM memories 
                         WHERE is_active = TRUE AND text ILIKE $1 AND agent_id = $2 AND user_id = $3"
                    )
                    .bind(&search_pattern)
                    .bind(agent_id.unwrap())
                    .bind(user_id);
                } else {
                    sql_query = sqlx::query(
                        "SELECT id, text, metadata FROM memories 
                         WHERE is_active = TRUE AND text ILIKE $1 AND user_id = $2"
                    )
                    .bind(&search_pattern)
                    .bind(user_id);
                }
            }

            let rows = sql_query.fetch_all(pool).await?;
            
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

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(results)
}

async fn handle_summarize(_args: SummarizeToolArgs) -> Result<Value> {
    // TODO: Implement LLM-based summarization
    // This should:
    Ok(json!({
        "content": [{
            "type": "text",
            "text": json!({
                "ok": true,
                "created": 0,
                "updated": 0,
                "message": "Summarization coming soon"
            })
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

    Ok(json!({
        "content": [{
            "type": "text",
            "text": json!({
                "ok": true,
                "deleted": deleted
            })
        }]
    }))
}

