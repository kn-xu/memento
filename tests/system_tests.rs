//! System tests for end-to-end MCP server workflows
//!
//! These tests simulate complete MCP tool call sequences as they would
//! occur in real usage, testing the full integration of all components.

use memento::database::{DatabaseClient, Memory, MemoryEvent};
use memento::embeddings::DummyEmbeddingProvider;
use memento::types::*;
use memento::vector_store::VectorStore;
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

/// Helper to setup a complete test environment matching MCP server setup
async fn setup_mcp_env() -> (DatabaseClient, Arc<VectorStore>) {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    (db, vector_store)
}

// ============================================================================
// memento.store System Tests
// ============================================================================

/// Simulates the complete memento.store tool call flow
async fn simulate_store(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: StoreToolArgs,
) -> Result<(String, String), String> {
    // Validate required fields (as MCP handler does)
    if args.agent_id.is_empty() {
        return Err("agent_id is required".to_string());
    }
    if args.text.is_empty() {
        return Err("text is required".to_string());
    }

    let event_type = args.event_type.unwrap_or_else(|| "user_msg".to_string());
    
    // Create event
    let event_id = Uuid::new_v4().to_string();
    let event = MemoryEvent {
        id: event_id.clone(),
        agent_id: args.agent_id.clone(),
        user_id: args.user_id.clone(),
        session_id: args.session_id.clone(),
        event_type: event_type.clone(),
        content: args.text.clone(),
        metadata: args.metadata.as_ref().map(|m| serde_json::to_string(m).unwrap()),
        created_at: Utc::now(),
        summarized_at: None,
    };
    db.insert_event(&event).await.map_err(|e| e.to_string())?;

    // Create memory
    let memory_id = Uuid::new_v4().to_string();
    let memory_type = match event_type.as_str() {
        "thought" => "working",
        _ => "episodic",
    };

    let memory = Memory {
        id: memory_id.clone(),
        agent_id: args.agent_id.clone(),
        user_id: args.user_id.clone(),
        session_id: args.session_id.clone(),
        memory_type: memory_type.to_string(),
        text: args.text.clone(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: Some(format!(r#"["{}"]"#, event_id)),
        metadata: args.metadata.as_ref().map(|m| serde_json::to_string(m).unwrap()),
        last_accessed_at: None,
        created_at: Utc::now(),
        expires_at: None,
    };
    db.insert_memory(&memory).await.map_err(|e| e.to_string())?;

    // Add embedding
    let mut meta = Metadata::new();
    meta.insert("agent_id".to_string(), serde_json::json!(&args.agent_id));
    if let Some(ref uid) = args.user_id {
        meta.insert("user_id".to_string(), serde_json::json!(uid));
    }
    if let Some(ref extra) = args.metadata {
        for (k, v) in extra {
            meta.insert(k.clone(), v.clone());
        }
    }
    vector_store
        .add(&memory_id, &args.text, None, meta)
        .await
        .map_err(|e| e.to_string())?;

    Ok((event_id, memory_id))
}

/// Simulates the complete memento.search tool call flow
async fn simulate_search(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: SearchToolArgs,
) -> Result<Vec<SearchResult>, String> {
    if args.agent_id.is_empty() || args.query.is_empty() {
        return Err("agent_id and query are required".to_string());
    }

    let k = args.k.unwrap_or(5) as usize;
    let filters = args.filters.unwrap_or_default();

    let vector_results = vector_store
        .search(&args.query, k, filters, Some(&args.agent_id), args.user_id.as_deref())
        .await
        .map_err(|e| e.to_string())?;

    // Fetch full memories
    let ids: Vec<String> = vector_results.iter().map(|r| r.memory_id.clone()).collect();
    let memories = db.get_memories_by_ids(&ids).await.map_err(|e| e.to_string())?;

    // Update access times
    for id in &ids {
        let _ = db.update_memory_access(id).await;
    }

    // Build results
    let results: Vec<SearchResult> = vector_results
        .iter()
        .filter_map(|vr| {
            memories.get(&vr.memory_id).map(|m| SearchResult {
                memory_id: m.id.clone(),
                text: m.text.clone(),
                score: vr.score,
                metadata: m.metadata.as_ref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default(),
            })
        })
        .collect();

    Ok(results)
}

/// Simulates the complete memento.forget tool call flow
async fn simulate_forget(
    db: &DatabaseClient,
    vector_store: &Arc<VectorStore>,
    args: ForgetToolArgs,
) -> Result<u32, String> {
    if args.agent_id.is_empty() {
        return Err("agent_id is required".to_string());
    }

    let mut forgotten = 0;

    if let Some(memory_id) = &args.memory_id {
        db.soft_delete_memory(memory_id).await.map_err(|e| e.to_string())?;
        vector_store.delete(memory_id).await.map_err(|e| e.to_string())?;
        forgotten = 1;
    } else if let Some(query) = &args.query {
        let results = vector_store
            .search(query, 10, Metadata::new(), Some(&args.agent_id), args.user_id.as_deref())
            .await
            .map_err(|e| e.to_string())?;

        for r in &results {
            db.soft_delete_memory(&r.memory_id).await.map_err(|e| e.to_string())?;
            vector_store.delete(&r.memory_id).await.map_err(|e| e.to_string())?;
            forgotten += 1;
        }
    }

    Ok(forgotten)
}

// ============================================================================
// Complete Workflow System Tests
// ============================================================================

#[tokio::test]
async fn test_system_store_search_forget_workflow() {
    let (db, vector_store) = setup_mcp_env().await;

    // Step 1: Store a preference
    let store_args = StoreToolArgs {
        text: "User preference: Always use 4-space indentation in Python".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        session_id: None,
        event_type: Some("preference".to_string()),
        metadata: Some({
            let mut m = Metadata::new();
            m.insert("tags".to_string(), serde_json::json!(["preference", "python", "formatting"]));
            m
        }),
    };
    let (event_id, memory_id) = simulate_store(&db, &vector_store, store_args).await.unwrap();
    assert!(!event_id.is_empty());
    assert!(!memory_id.is_empty());

    // Step 2: Search for the preference
    let search_args = SearchToolArgs {
        query: "Python indentation".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        k: Some(5),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].text.contains("4-space indentation"));

    // Step 3: Forget the preference
    let forget_args = ForgetToolArgs {
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        memory_id: Some(memory_id.clone()),
        query: None,
    };
    let forgotten = simulate_forget(&db, &vector_store, forget_args).await.unwrap();
    assert_eq!(forgotten, 1);

    // Step 4: Verify it's gone
    let search_args = SearchToolArgs {
        query: "Python indentation".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        k: Some(5),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_system_multi_user_isolation() {
    let (db, vector_store) = setup_mcp_env().await;

    // Alice stores her preference
    let alice_store = StoreToolArgs {
        text: "I prefer dark mode themes".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        session_id: None,
        event_type: Some("preference".to_string()),
        metadata: None,
    };
    simulate_store(&db, &vector_store, alice_store).await.unwrap();

    // Bob stores his preference
    let bob_store = StoreToolArgs {
        text: "I prefer light mode themes".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("bob".to_string()),
        session_id: None,
        event_type: Some("preference".to_string()),
        metadata: None,
    };
    simulate_store(&db, &vector_store, bob_store).await.unwrap();

    // Alice searches - should only see her preference
    let alice_search = SearchToolArgs {
        query: "theme preference".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("alice".to_string()),
        k: Some(10),
        filters: None,
    };
    let alice_results = simulate_search(&db, &vector_store, alice_search).await.unwrap();
    assert_eq!(alice_results.len(), 1);
    assert!(alice_results[0].text.contains("dark mode"));

    // Bob searches - should only see his preference
    let bob_search = SearchToolArgs {
        query: "theme preference".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("bob".to_string()),
        k: Some(10),
        filters: None,
    };
    let bob_results = simulate_search(&db, &vector_store, bob_search).await.unwrap();
    assert_eq!(bob_results.len(), 1);
    assert!(bob_results[0].text.contains("light mode"));
}

#[tokio::test]
async fn test_system_bulk_store_and_search() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store 20 different preferences
    let preferences = vec![
        "Use Result<T, E> for error handling",
        "Prefer async/await over callbacks",
        "Always add doc comments to public functions",
        "Use snake_case for variables",
        "Prefer match over if-let chains",
        "Keep functions under 50 lines",
        "Write unit tests for all public APIs",
        "Use semantic versioning",
        "Prefer composition over inheritance",
        "Keep dependencies minimal",
        "Use type aliases for complex types",
        "Prefer iterators over manual loops",
        "Document all public modules",
        "Use clippy for linting",
        "Format with rustfmt",
        "Prefer owned types in APIs",
        "Use thiserror for custom errors",
        "Prefer early returns",
        "Group related imports",
        "Use descriptive variable names",
    ];

    for pref in &preferences {
        let args = StoreToolArgs {
            text: format!("Coding preference: {}", pref),
            agent_id: "cursor".to_string(),
            user_id: Some("dev".to_string()),
            session_id: None,
            event_type: Some("preference".to_string()),
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Search for error handling
    let search_args = SearchToolArgs {
        query: "error handling".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        k: Some(5),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert!(!results.is_empty());
    // Should find the error handling preference
    assert!(results.iter().any(|r| r.text.contains("Result<T, E>") || r.text.contains("thiserror")));
}

#[tokio::test]
async fn test_system_forget_by_query() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store several related preferences
    for topic in ["tabs vs spaces", "indentation style", "formatting rules"] {
        let args = StoreToolArgs {
            text: format!("Formatting preference about: {}", topic),
            agent_id: "cursor".to_string(),
            user_id: Some("user".to_string()),
            session_id: None,
            event_type: Some("preference".to_string()),
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Forget all formatting preferences by query
    let forget_args = ForgetToolArgs {
        agent_id: "cursor".to_string(),
        user_id: Some("user".to_string()),
        memory_id: None,
        query: Some("formatting".to_string()),
    };
    let forgotten = simulate_forget(&db, &vector_store, forget_args).await.unwrap();
    assert!(forgotten >= 3); // Should forget all 3

    // Verify they're gone
    let search_args = SearchToolArgs {
        query: "formatting".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("user".to_string()),
        k: Some(10),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_system_session_scoped_memories() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store memories in different sessions
    let session1_args = StoreToolArgs {
        text: "Working on authentication module".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        session_id: Some("session-auth".to_string()),
        event_type: Some("context".to_string()),
        metadata: None,
    };
    simulate_store(&db, &vector_store, session1_args).await.unwrap();

    let session2_args = StoreToolArgs {
        text: "Working on database module".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        session_id: Some("session-db".to_string()),
        event_type: Some("context".to_string()),
        metadata: None,
    };
    simulate_store(&db, &vector_store, session2_args).await.unwrap();

    // Both should be searchable (session filtering is optional)
    let search_args = SearchToolArgs {
        query: "working on".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        k: Some(10),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert_eq!(results.len(), 2);
}

// ============================================================================
// Error Handling System Tests
// ============================================================================

#[tokio::test]
async fn test_system_store_validation_errors() {
    let (db, vector_store) = setup_mcp_env().await;

    // Missing agent_id
    let args = StoreToolArgs {
        text: "Some text".to_string(),
        agent_id: "".to_string(),
        user_id: None,
        session_id: None,
        event_type: None,
        metadata: None,
    };
    let result = simulate_store(&db, &vector_store, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("agent_id"));

    // Missing text
    let args = StoreToolArgs {
        text: "".to_string(),
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        event_type: None,
        metadata: None,
    };
    let result = simulate_store(&db, &vector_store, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("text"));
}

#[tokio::test]
async fn test_system_search_validation_errors() {
    let (db, vector_store) = setup_mcp_env().await;

    // Missing agent_id
    let args = SearchToolArgs {
        query: "test".to_string(),
        agent_id: "".to_string(),
        user_id: None,
        k: None,
        filters: None,
    };
    let result = simulate_search(&db, &vector_store, args).await;
    assert!(result.is_err());

    // Missing query
    let args = SearchToolArgs {
        query: "".to_string(),
        agent_id: "cursor".to_string(),
        user_id: None,
        k: None,
        filters: None,
    };
    let result = simulate_search(&db, &vector_store, args).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_system_forget_validation_errors() {
    let (db, vector_store) = setup_mcp_env().await;

    // Missing agent_id
    let args = ForgetToolArgs {
        agent_id: "".to_string(),
        user_id: None,
        memory_id: Some("mem-1".to_string()),
        query: None,
    };
    let result = simulate_forget(&db, &vector_store, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("agent_id"));
}

// ============================================================================
// Metadata System Tests
// ============================================================================

#[tokio::test]
async fn test_system_store_with_rich_metadata() {
    let (db, vector_store) = setup_mcp_env().await;

    let mut metadata = Metadata::new();
    metadata.insert("tags".to_string(), serde_json::json!(["rust", "programming", "best-practices"]));
    metadata.insert("project".to_string(), serde_json::json!("memento"));
    metadata.insert("priority".to_string(), serde_json::json!(1));
    metadata.insert("nested".to_string(), serde_json::json!({"key": "value", "arr": [1, 2, 3]}));

    let args = StoreToolArgs {
        text: "Complex metadata test".to_string(),
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        event_type: None,
        metadata: Some(metadata),
    };
    
    let (_, memory_id) = simulate_store(&db, &vector_store, args).await.unwrap();
    
    // Retrieve and verify metadata
    let memory = db.get_memory(&memory_id).await.unwrap().unwrap();
    let stored_meta: Metadata = serde_json::from_str(memory.metadata.as_ref().unwrap()).unwrap();
    
    assert!(stored_meta.contains_key("tags"));
    assert!(stored_meta.contains_key("nested"));
}


// ============================================================================
// Summarization System Tests
// ============================================================================

/// Simulates memento.summarize - returns unsummarized events
async fn simulate_summarize(
    db: &DatabaseClient,
    args: SummarizeToolArgs,
) -> Result<Vec<MemoryEvent>, String> {
    if args.agent_id.is_empty() {
        return Err("agent_id is required".to_string());
    }

    let limit = args.limit.unwrap_or(50);
    
    let events = db.list_unsummarized_events(
        &args.agent_id,
        args.user_id.as_deref(),
        args.session_id.as_deref(),
        limit,
    ).await.map_err(|e| e.to_string())?;

    Ok(events)
}

/// Simulates memento.mark_summarized - marks events as processed
async fn simulate_mark_summarized(
    db: &DatabaseClient,
    args: MarkSummarizedToolArgs,
) -> Result<usize, String> {
    if args.agent_id.is_empty() {
        return Err("agent_id is required".to_string());
    }
    if args.event_ids.is_empty() {
        return Err("event_ids cannot be empty".to_string());
    }

    db.mark_events_summarized(&args.agent_id, &args.event_ids)
        .await
        .map_err(|e| e.to_string())?;

    Ok(args.event_ids.len())
}

#[tokio::test]
async fn test_system_summarize_returns_unsummarized_events() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store several events
    for i in 1..=5 {
        let args = StoreToolArgs {
            text: format!("Event content number {}", i),
            agent_id: "cursor".to_string(),
            user_id: Some("user".to_string()),
            session_id: None,
            event_type: Some("user_msg".to_string()),
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Get unsummarized events
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: Some("user".to_string()),
        session_id: None,
        limit: Some(50),
    };
    let events = simulate_summarize(&db, summarize_args).await.unwrap();

    assert_eq!(events.len(), 5);
    assert!(events.iter().any(|e| e.content.contains("Event content")));
}

#[tokio::test]
async fn test_system_summarize_with_no_events() {
    let (db, _) = setup_mcp_env().await;

    // No events stored
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        limit: None,
    };
    let events = simulate_summarize(&db, summarize_args).await.unwrap();

    assert!(events.is_empty());
}

#[tokio::test]
async fn test_system_summarize_respects_limit() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store 10 events
    for i in 1..=10 {
        let args = StoreToolArgs {
            text: format!("Event {}", i),
            agent_id: "cursor".to_string(),
            user_id: None,
            session_id: None,
            event_type: None,
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Request only 3
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        limit: Some(3),
    };
    let events = simulate_summarize(&db, summarize_args).await.unwrap();

    assert_eq!(events.len(), 3);
}

#[tokio::test]
async fn test_system_mark_summarized_removes_from_queue() {
    let (db, vector_store) = setup_mcp_env().await;

    // Store events
    let mut event_ids = Vec::new();
    for i in 1..=5 {
        let args = StoreToolArgs {
            text: format!("Event to summarize {}", i),
            agent_id: "cursor".to_string(),
            user_id: None,
            session_id: None,
            event_type: None,
            metadata: None,
        };
        let (event_id, _) = simulate_store(&db, &vector_store, args).await.unwrap();
        event_ids.push(event_id);
    }

    // Verify all are unsummarized
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        limit: None,
    };
    let events = simulate_summarize(&db, summarize_args.clone()).await.unwrap();
    assert_eq!(events.len(), 5);

    // Mark first 3 as summarized
    let mark_args = MarkSummarizedToolArgs {
        agent_id: "cursor".to_string(),
        event_ids: event_ids[0..3].to_vec(),
    };
    let marked = simulate_mark_summarized(&db, mark_args).await.unwrap();
    assert_eq!(marked, 3);

    // Only 2 should remain unsummarized
    let events = simulate_summarize(&db, summarize_args).await.unwrap();
    assert_eq!(events.len(), 2);
}

#[tokio::test]
async fn test_system_full_summarization_workflow() {
    let (db, vector_store) = setup_mcp_env().await;

    // Step 1: User interactions create events
    let interactions = vec![
        "I prefer using async/await over callbacks",
        "Always use 4-space indentation",
        "Database should use PostgreSQL for production",
    ];

    let mut all_event_ids = Vec::new();
    for text in interactions {
        let args = StoreToolArgs {
            text: text.to_string(),
            agent_id: "cursor".to_string(),
            user_id: Some("dev".to_string()),
            session_id: Some("session-1".to_string()),
            event_type: Some("preference".to_string()),
            metadata: None,
        };
        let (event_id, _) = simulate_store(&db, &vector_store, args).await.unwrap();
        all_event_ids.push(event_id);
    }

    // Step 2: Summarize retrieves unsummarized events
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        session_id: Some("session-1".to_string()),
        limit: None,
    };
    let events = simulate_summarize(&db, summarize_args).await.unwrap();
    assert_eq!(events.len(), 3);

    // Step 3: LLM would analyze and store a summary
    let summary_args = StoreToolArgs {
        text: "Developer preferences: async/await, 4-space indent, PostgreSQL for prod".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        session_id: None,
        event_type: Some("summary".to_string()),
        metadata: Some({
            let mut m = Metadata::new();
            m.insert("summarized_from".to_string(), serde_json::json!(all_event_ids));
            m
        }),
    };
    simulate_store(&db, &vector_store, summary_args).await.unwrap();

    // Step 4: Mark original events as summarized
    let mark_args = MarkSummarizedToolArgs {
        agent_id: "cursor".to_string(),
        event_ids: all_event_ids.clone(),
    };
    simulate_mark_summarized(&db, mark_args).await.unwrap();

    // Step 5: Verify no unsummarized events remain
    let summarize_args = SummarizeToolArgs {
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        session_id: Some("session-1".to_string()),
        limit: None,
    };
    let events = simulate_summarize(&db, summarize_args).await.unwrap();
    assert!(events.is_empty());

    // Step 6: The summary should be searchable
    let search_args = SearchToolArgs {
        query: "developer preferences async".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("dev".to_string()),
        k: Some(5),
        filters: None,
    };
    let results = simulate_search(&db, &vector_store, search_args).await.unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.text.contains("4-space indent") || r.text.contains("PostgreSQL")));
}

#[tokio::test]
async fn test_system_summarize_agent_isolation() {
    let (db, vector_store) = setup_mcp_env().await;

    // Agent A stores events
    for i in 1..=3 {
        let args = StoreToolArgs {
            text: format!("Agent A event {}", i),
            agent_id: "agent-a".to_string(),
            user_id: None,
            session_id: None,
            event_type: None,
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Agent B stores events
    for i in 1..=2 {
        let args = StoreToolArgs {
            text: format!("Agent B event {}", i),
            agent_id: "agent-b".to_string(),
            user_id: None,
            session_id: None,
            event_type: None,
            metadata: None,
        };
        simulate_store(&db, &vector_store, args).await.unwrap();
    }

    // Agent A should only see their events
    let summarize_a = SummarizeToolArgs {
        agent_id: "agent-a".to_string(),
        user_id: None,
        session_id: None,
        limit: None,
    };
    let events_a = simulate_summarize(&db, summarize_a).await.unwrap();
    assert_eq!(events_a.len(), 3);
    assert!(events_a.iter().all(|e| e.agent_id == "agent-a"));

    // Agent B should only see their events
    let summarize_b = SummarizeToolArgs {
        agent_id: "agent-b".to_string(),
        user_id: None,
        session_id: None,
        limit: None,
    };
    let events_b = simulate_summarize(&db, summarize_b).await.unwrap();
    assert_eq!(events_b.len(), 2);
    assert!(events_b.iter().all(|e| e.agent_id == "agent-b"));
}

#[tokio::test]
async fn test_system_mark_summarized_validation_errors() {
    let (db, _) = setup_mcp_env().await;

    // Missing agent_id
    let args = MarkSummarizedToolArgs {
        agent_id: "".to_string(),
        event_ids: vec!["evt-1".to_string()],
    };
    let result = simulate_mark_summarized(&db, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("agent_id"));

    // Empty event_ids
    let args = MarkSummarizedToolArgs {
        agent_id: "cursor".to_string(),
        event_ids: vec![],
    };
    let result = simulate_mark_summarized(&db, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("event_ids"));
}

#[tokio::test]
async fn test_system_summarize_validation_errors() {
    let (db, _) = setup_mcp_env().await;

    // Missing agent_id
    let args = SummarizeToolArgs {
        agent_id: "".to_string(),
        user_id: None,
        session_id: None,
        limit: None,
    };
    let result = simulate_summarize(&db, args).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("agent_id"));
}
