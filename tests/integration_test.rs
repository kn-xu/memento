//! Integration tests for Memento memory engine

use memento::database::{DatabaseClient, Memory, MemoryEvent};
use memento::embeddings::{DummyEmbeddingProvider, EmbeddingProvider};
use memento::vector_store::VectorStore;
use memento::types::Metadata;
use chrono::Utc;
use std::sync::Arc;

/// Helper to create an in-memory test database with vector store
async fn setup_test_env() -> (DatabaseClient, Arc<VectorStore>) {
    let db = DatabaseClient::new("sqlite::memory:").await.unwrap();
    let provider: Box<dyn EmbeddingProvider + Send + Sync> = 
        Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    (db, vector_store)
}

#[tokio::test]
async fn test_full_store_and_search_flow() {
    let (db, vector_store) = setup_test_env().await;

    // Simulate storing a memory (like memento.store does)
    let event = MemoryEvent {
        id: "evt-integration-1".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: None,
        event_type: "user_msg".to_string(),
        content: "I prefer using async/await over callbacks in JavaScript".to_string(),
        metadata: None,
        created_at: Utc::now(),
        summarized_at: None,
    };
    db.insert_event(&event).await.unwrap();

    // Create memory with embedding
    let memory = Memory {
        id: "mem-integration-1".to_string(),
        agent_id: "cursor".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "I prefer using async/await over callbacks in JavaScript".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: Some(r#"["evt-integration-1"]"#.to_string()),
        metadata: None,
        last_accessed_at: None,
        created_at: Utc::now(),
        expires_at: None,
    };
    db.insert_memory(&memory).await.unwrap();

    let mut meta = Metadata::new();
    meta.insert("agent_id".to_string(), serde_json::json!("cursor"));
    meta.insert("user_id".to_string(), serde_json::json!("user-1"));
    vector_store.add(
        "mem-integration-1",
        "I prefer using async/await over callbacks in JavaScript",
        None,
        meta,
    ).await.unwrap();

    // Search for the memory
    let results = vector_store.search(
        "async await JavaScript preference",
        5,
        Metadata::new(),
        Some("cursor"),
        Some("user-1"),
    ).await.unwrap();

    assert!(!results.is_empty());
    
    // Verify we can retrieve the full memory
    let found_memory = db.get_memory(&results[0].memory_id).await.unwrap();
    assert!(found_memory.is_some());
    assert!(found_memory.unwrap().text.contains("async/await"));
}

#[tokio::test]
async fn test_multi_agent_isolation() {
    let (db, vector_store) = setup_test_env().await;

    // Store memories for two different agents
    for (agent, text) in [
        ("agent-a", "Agent A's secret preference"),
        ("agent-b", "Agent B's secret preference"),
    ] {
        let memory = Memory {
            id: format!("mem-{}", agent),
            agent_id: agent.to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: text.to_string(),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: None,
            metadata: None,
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
        };
        db.insert_memory(&memory).await.unwrap();

        let mut meta = Metadata::new();
        meta.insert("agent_id".to_string(), serde_json::json!(agent));
        vector_store.add(&format!("mem-{}", agent), text, None, meta).await.unwrap();
    }

    // Agent A should only see their own memories
    let results_a = vector_store.search(
        "secret preference",
        10,
        Metadata::new(),
        Some("agent-a"),
        None,
    ).await.unwrap();

    assert_eq!(results_a.len(), 1);
    assert_eq!(results_a[0].memory_id, "mem-agent-a");

    // Agent B should only see their own memories
    let results_b = vector_store.search(
        "secret preference",
        10,
        Metadata::new(),
        Some("agent-b"),
        None,
    ).await.unwrap();

    assert_eq!(results_b.len(), 1);
    assert_eq!(results_b[0].memory_id, "mem-agent-b");
}

#[tokio::test]
async fn test_forget_flow() {
    let (db, vector_store) = setup_test_env().await;

    // Store a memory
    let memory = Memory {
        id: "mem-to-forget".to_string(),
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "My phone number is 555-1234".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: Utc::now(),
        expires_at: None,
    };
    db.insert_memory(&memory).await.unwrap();

    let mut meta = Metadata::new();
    meta.insert("agent_id".to_string(), serde_json::json!("cursor"));
    vector_store.add("mem-to-forget", "My phone number is 555-1234", None, meta).await.unwrap();

    // Verify it exists
    let results_before = vector_store.search(
        "phone number",
        5,
        Metadata::new(),
        Some("cursor"),
        None,
    ).await.unwrap();
    assert!(!results_before.is_empty());

    // Forget it (soft delete + remove embedding)
    db.soft_delete_memory("mem-to-forget").await.unwrap();
    vector_store.delete("mem-to-forget").await.unwrap();

    // Verify it's no longer searchable
    let results_after = vector_store.search(
        "phone number",
        5,
        Metadata::new(),
        Some("cursor"),
        None,
    ).await.unwrap();
    assert!(results_after.is_empty());

    // Memory still exists but is inactive
    let memory = db.get_memory("mem-to-forget").await.unwrap().unwrap();
    assert!(!memory.is_active);
}

#[tokio::test]
async fn test_event_summarization_tracking() {
    let (db, _) = setup_test_env().await;

    // Insert multiple events
    for i in 1..=5 {
        let event = MemoryEvent {
            id: format!("evt-sum-{}", i),
            agent_id: "cursor".to_string(),
            user_id: Some("user-1".to_string()),
            session_id: Some("session-1".to_string()),
            event_type: "user_msg".to_string(),
            content: format!("Event number {}", i),
            metadata: None,
            created_at: Utc::now(),
            summarized_at: None,
        };
        db.insert_event(&event).await.unwrap();
    }

    // All events should be unsummarized
    let unsummarized = db.list_unsummarized_events(
        "cursor",
        Some("user-1"),
        Some("session-1"),
        100,
    ).await.unwrap();
    assert_eq!(unsummarized.len(), 5);

    // Mark some as summarized
    db.mark_events_summarized(
        "cursor",
        &["evt-sum-1".to_string(), "evt-sum-2".to_string(), "evt-sum-3".to_string()],
    ).await.unwrap();

    // Only 2 should remain unsummarized
    let unsummarized = db.list_unsummarized_events(
        "cursor",
        Some("user-1"),
        Some("session-1"),
        100,
    ).await.unwrap();
    assert_eq!(unsummarized.len(), 2);
    assert!(unsummarized.iter().any(|e| e.id == "evt-sum-4"));
    assert!(unsummarized.iter().any(|e| e.id == "evt-sum-5"));
}

#[tokio::test]
async fn test_user_isolation_within_agent() {
    let (db, vector_store) = setup_test_env().await;

    // Store memories for different users under the same agent
    for (user, text) in [
        ("user-alice", "Alice prefers vim"),
        ("user-bob", "Bob prefers emacs"),
    ] {
        let memory = Memory {
            id: format!("mem-{}", user),
            agent_id: "cursor".to_string(),
            user_id: Some(user.to_string()),
            session_id: None,
            memory_type: "preference".to_string(),
            text: text.to_string(),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: None,
            metadata: None,
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
        };
        db.insert_memory(&memory).await.unwrap();

        let mut meta = Metadata::new();
        meta.insert("agent_id".to_string(), serde_json::json!("cursor"));
        meta.insert("user_id".to_string(), serde_json::json!(user));
        vector_store.add(&format!("mem-{}", user), text, None, meta).await.unwrap();
    }

    // Search for Alice's preferences
    let alice_results = vector_store.search(
        "editor preference",
        10,
        Metadata::new(),
        Some("cursor"),
        Some("user-alice"),
    ).await.unwrap();

    assert_eq!(alice_results.len(), 1);
    assert_eq!(alice_results[0].memory_id, "mem-user-alice");

    // Search for Bob's preferences
    let bob_results = vector_store.search(
        "editor preference",
        10,
        Metadata::new(),
        Some("cursor"),
        Some("user-bob"),
    ).await.unwrap();

    assert_eq!(bob_results.len(), 1);
    assert_eq!(bob_results[0].memory_id, "mem-user-bob");
}

#[tokio::test]
async fn test_memory_access_tracking() {
    let (db, _) = setup_test_env().await;

    let memory = Memory {
        id: "mem-access-track".to_string(),
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Frequently accessed memory".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: Utc::now(),
        expires_at: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Initially no access time
    let mem = db.get_memory("mem-access-track").await.unwrap().unwrap();
    assert!(mem.last_accessed_at.is_none());

    // Simulate access
    db.update_memory_access("mem-access-track").await.unwrap();

    // Now has access time
    let mem = db.get_memory("mem-access-track").await.unwrap().unwrap();
    assert!(mem.last_accessed_at.is_some());
}

#[tokio::test]
async fn test_batch_memory_retrieval() {
    let (db, _) = setup_test_env().await;

    // Insert 10 memories
    for i in 1..=10 {
        let memory = Memory {
            id: format!("mem-batch-{}", i),
            agent_id: "cursor".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Batch memory {}", i),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: None,
            metadata: None,
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
        };
        db.insert_memory(&memory).await.unwrap();
    }

    // Retrieve subset by IDs
    let ids: Vec<String> = (1..=5).map(|i| format!("mem-batch-{}", i)).collect();
    let memories = db.get_memories_by_ids(&ids).await.unwrap();

    assert_eq!(memories.len(), 5);
    for i in 1..=5 {
        assert!(memories.contains_key(&format!("mem-batch-{}", i)));
    }
    for i in 6..=10 {
        assert!(!memories.contains_key(&format!("mem-batch-{}", i)));
    }
}

