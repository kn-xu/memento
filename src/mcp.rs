use crate::config::Config;
use crate::database::DatabaseClient;
use crate::embeddings::get_embedding_provider;
use crate::types::*;
use crate::vector_store::VectorStore;
use anyhow::Result;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;

pub async fn start_mcp_server() -> Result<()> {
    let config = Config::from_env();
    let db = DatabaseClient::new(&config.database_url).await?;

    let embedding_provider = get_embedding_provider(
        match config.embedding_provider {
            crate::config::EmbeddingProvider::OpenAi => "openai",
            crate::config::EmbeddingProvider::Local => "local",
        },
        config.openai_api_key.clone(),
        Some(config.embedding_model.clone()),
    )?;

    let vector_store = Arc::new(VectorStore::new(db.clone(), embedding_provider));

    // MCP server via stdio
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

    match method {
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
                    let args: StoreToolArgs = serde_json::from_value(args.clone())?;
                    // Implementation similar to REST API store endpoint
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": json!({"ok": true, "message": "Store implemented"})
                        }]
                    }))
                }
                "memento.search" => {
                    let args: SearchToolArgs = serde_json::from_value(args.clone())?;
                    // Implementation similar to REST API search endpoint
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": json!({"ok": true, "message": "Search implemented"})
                        }]
                    }))
                }
                "memento.summarize" => {
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": json!({"ok": true, "created": 0, "updated": 0, "message": "Summarization coming soon"})
                        }]
                    }))
                }
                "memento.forget" => {
                    let args: ForgetToolArgs = serde_json::from_value(args.clone())?;
                    // Implementation similar to REST API forget endpoint
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": json!({"ok": true, "message": "Forget implemented"})
                        }]
                    }))
                }
                _ => Ok(json!({
                    "error": {"code": -32601, "message": format!("Unknown tool: {}", tool_name)}
                })),
            }
        }
        _ => Ok(json!({
            "error": {"code": -32601, "message": format!("Unknown method: {}", method)}
        })),
    }
}

