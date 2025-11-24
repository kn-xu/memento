use crate::database::{DatabaseClient, Memory, MemoryEvent};
use crate::embeddings::EmbeddingProvider;
use crate::types::*;
use crate::vector_store::VectorStore;
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use chrono::Utc;
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

pub struct AppState {
    pub db: DatabaseClient,
    pub vector_store: Arc<VectorStore>,
}

pub async fn start_server(
    db: DatabaseClient,
    embedding_provider: Box<dyn EmbeddingProvider + Send + Sync>,
    host: String,
    port: u16,
) -> Result<()> {
    let vector_store = Arc::new(VectorStore::new(db.clone(), embedding_provider));
    
    let app_state = Arc::new(AppState {
        db,
        vector_store,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/memory/store", post(store_memory))
        .route("/memory/search", post(search_memory))
        .route("/memory/summarize", post(summarize_memory))
        .route("/memory/forget", post(forget_memory))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
    println!("🚀 Memento server running on http://{}:{}", host, port);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn store_memory(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<StoreRequest>,
) -> Result<Json<StoreResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.agent_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "agent_id is required".to_string(),
            }),
        ));
    }

    let text = payload.text.or(payload.content).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text or content is required".to_string(),
            }),
        )
    })?;

    let event_id = Uuid::new_v4().to_string();
    let event = MemoryEvent {
        id: event_id.clone(),
        agent_id: payload.agent_id.clone(),
        user_id: payload.user_id.clone(),
        session_id: payload.session_id.clone(),
        event_type: payload.event_type.clone().unwrap_or_else(|| "user_msg".to_string()),
        content: text.clone(),
        metadata: payload
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default()),
        created_at: Utc::now(),
        summarized_at: None,
    };

    state.db.insert_event(&event).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let event_type = payload.event_type.as_deref().unwrap_or("user_msg");
    let mut memory_id: Option<String> = None;
    if text.len() > 10
        && matches!(event_type, "user_msg" | "thought")
    {
        let memory = Memory {
            id: Uuid::new_v4().to_string(),
            agent_id: payload.agent_id.clone(),
            user_id: payload.user_id.clone(),
            session_id: payload.session_id.clone(),
            memory_type: "episodic".to_string(),
            text: text.clone(),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: Some(json!([event_id]).to_string()),
            metadata: payload
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).unwrap_or_default()),
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
        };

        let mem_id = memory.id.clone();
        state.db.insert_memory(&memory).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

        let mut vector_metadata = Metadata::new();
        vector_metadata.insert("agent_id".to_string(), json!(payload.agent_id));
        if let Some(user_id) = &payload.user_id {
            vector_metadata.insert("user_id".to_string(), json!(user_id));
        }
        if let Some(session_id) = &payload.session_id {
            vector_metadata.insert("session_id".to_string(), json!(session_id));
        }
        vector_metadata.insert("memory_type".to_string(), json!("episodic"));
        if let Some(metadata) = &payload.metadata {
            for (k, v) in metadata {
                vector_metadata.insert(k.clone(), v.clone());
            }
        }

        if let Err(e) = state.vector_store.add(&mem_id, &text, None, vector_metadata).await {
            let _ = state.db.soft_delete_memory(&mem_id).await;
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to store embedding: {}", e),
                }),
            ));
        }

        memory_id = Some(mem_id);
    }

    Ok(Json(StoreResponse {
        ok: true,
        event_id,
        memory_id,
    }))
}

async fn search_memory(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.agent_id.is_empty() || payload.query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "agent_id and query are required".to_string(),
            }),
        ));
    }

    let vector_results = state
        .vector_store
        .search(
            &payload.query,
            payload.k,
            payload.filters.unwrap_or_default(),
            Some(&payload.agent_id),
            payload.user_id.as_deref(),
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let mut results = Vec::new();
    for result in vector_results {
        if let Some(memory) = state.db.get_memory(&result.memory_id).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })? {
            if memory.is_active {
                state
                    .db
                    .update_memory_access(&memory.id)
                    .await
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: e.to_string(),
                            }),
                        )
                    })?;

                let mut metadata = if let Some(meta_str) = &memory.metadata {
                    serde_json::from_str(meta_str).unwrap_or_default()
                } else {
                    Metadata::new()
                };

                for (k, v) in result.metadata {
                    metadata.insert(k, v);
                }

                results.push(SearchResult {
                    memory_id: memory.id,
                    text: memory.text,
                    score: result.score,
                    metadata,
                });
            }
        }
    }

    Ok(Json(SearchResponse { ok: true, results }))
}

async fn summarize_memory(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<SummarizeRequest>,
) -> Result<Json<SummarizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.agent_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "agent_id is required".to_string(),
            }),
        ));
    }

    Ok(Json(SummarizeResponse {
        ok: true,
        created: 0,
        updated: 0,
        message: Some("Summarization coming soon".to_string()),
    }))
}

async fn forget_memory(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ForgetRequest>,
) -> Result<Json<ForgetResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.agent_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "agent_id is required".to_string(),
            }),
        ));
    }

    let mut deleted = 0;

    if let Some(memory_id) = payload.memory_id {
        if let Some(memory) = state.db.get_memory(&memory_id).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })? {
            if memory.agent_id == payload.agent_id {
                state
                    .db
                    .soft_delete_memory(&memory_id)
                    .await
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: e.to_string(),
                            }),
                        )
                    })?;
                state
                    .vector_store
                    .delete(&memory_id)
                    .await
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: e.to_string(),
                            }),
                        )
                    })?;
                deleted = 1;
            }
        }
    } else if let Some(query) = payload.query {
        let vector_results = state
            .vector_store
            .search(
                &query,
                20,
                Metadata::new(),
                Some(&payload.agent_id),
                payload.user_id.as_deref(),
            )
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

        for result in vector_results {
            if let Some(memory) = state.db.get_memory(&result.memory_id).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })? {
                if memory.is_active {
                    state
                        .db
                        .soft_delete_memory(&memory.id)
                        .await
                        .map_err(|e| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse {
                                    error: e.to_string(),
                                }),
                            )
                        })?;
                    state
                        .vector_store
                        .delete(&memory.id)
                        .await
                        .map_err(|e| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse {
                                    error: e.to_string(),
                                }),
                            )
                        })?;
                    deleted += 1;
                }
            }
        }
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Either memory_id or query is required".to_string(),
            }),
        ));
    }

    Ok(Json(ForgetResponse { ok: true, deleted }))
}

