//! Functional tests for edge cases and error handling
//!
//! These tests verify individual component behavior under various conditions,
//! including error cases, boundary conditions, and edge cases.

use memento::database::{DatabaseClient, Memory, MemoryEvent};
use memento::embeddings::DummyEmbeddingProvider;
use memento::types::Metadata;
use memento::vector_store::VectorStore;
use chrono::Utc;
use std::sync::Arc;

// ============================================================================
// Database Edge Cases
// ============================================================================

#[tokio::test]
async fn test_insert_duplicate_event_id_fails() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let event = MemoryEvent {
        id: "dup-event".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        event_type: "test".to_string(),
        content: "First event".to_string(),
        metadata: None,
        created_at: Utc::now(),
        summarized_at: None,
    };
    
    // First insert succeeds
    db.insert_event(&event).await.unwrap();
    
    // Second insert with same ID should fail
    let result = db.insert_event(&event).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_insert_duplicate_memory_id_fails() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let memory = Memory {
        id: "dup-mem".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "First memory".to_string(),
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
    let result = db.insert_memory(&memory).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_get_nonexistent_memory_returns_none() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let result = db.get_memory("nonexistent-id").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_soft_delete_nonexistent_memory_succeeds() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Should not error even if memory doesn't exist
    let result = db.soft_delete_memory("nonexistent-id").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_update_access_nonexistent_memory() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Should not error even if memory doesn't exist
    let result = db.update_memory_access("nonexistent-id").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_empty_batch_retrieval() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let ids: Vec<String> = vec![];
    let memories = db.get_memories_by_ids(&ids).await.unwrap();
    assert!(memories.is_empty());
}

#[tokio::test]
async fn test_batch_retrieval_with_nonexistent_ids() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Insert one real memory
    let memory = Memory {
        id: "real-mem".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Real memory".to_string(),
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
    
    // Request mix of real and fake IDs
    let ids = vec![
        "fake-1".to_string(),
        "real-mem".to_string(),
        "fake-2".to_string(),
    ];
    let memories = db.get_memories_by_ids(&ids).await.unwrap();
    
    assert_eq!(memories.len(), 1);
    assert!(memories.contains_key("real-mem"));
}

#[tokio::test]
async fn test_list_unsummarized_with_no_events() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let events = db.list_unsummarized_events("agent", None, None, 100).await.unwrap();
    assert!(events.is_empty());
}

#[tokio::test]
async fn test_memory_with_all_optional_fields() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let mut metadata = Metadata::new();
    metadata.insert("key".to_string(), serde_json::json!("value"));
    
    // First insert a memory that can be superseded
    let old_memory = Memory {
        id: "old-mem".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Old memory".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: Utc::now(),
        expires_at: None,
    };
    db.insert_memory(&old_memory).await.unwrap();
    
    let memory = Memory {
        id: "full-mem".to_string(),
        agent_id: "test".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: Some("session-1".to_string()),
        memory_type: "episodic".to_string(),
        text: "Full memory".to_string(),
        importance: 0.9,
        is_active: true,
        supersedes_id: Some("old-mem".to_string()),
        source_event_ids: Some(r#"["evt-1", "evt-2"]"#.to_string()),
        metadata: Some(serde_json::to_string(&metadata).unwrap()),
        last_accessed_at: Some(Utc::now()),
        created_at: Utc::now(),
        expires_at: Some(Utc::now() + chrono::Duration::days(30)),
    };
    
    db.insert_memory(&memory).await.unwrap();
    
    let retrieved = db.get_memory("full-mem").await.unwrap().unwrap();
    assert_eq!(retrieved.user_id, Some("user-1".to_string()));
    assert_eq!(retrieved.session_id, Some("session-1".to_string()));
    assert_eq!(retrieved.supersedes_id, Some("old-mem".to_string()));
    assert!(retrieved.last_accessed_at.is_some());
    assert!(retrieved.expires_at.is_some());
}

// ============================================================================
// Vector Store Edge Cases
// ============================================================================

#[tokio::test]
async fn test_search_empty_database() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db, provider));
    
    let results = vector_store
        .search("anything", 10, Metadata::new(), Some("agent"), None)
        .await
        .unwrap();
    
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_with_k_zero() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    
    // Insert a memory
    let memory = Memory {
        id: "mem-k0".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Test".to_string(),
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
    vector_store.add("mem-k0", "Test", None, Metadata::new()).await.unwrap();
    
    // Search with k=0
    let results = vector_store
        .search("Test", 0, Metadata::new(), Some("test"), None)
        .await
        .unwrap();
    
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_with_large_k() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    
    // Insert just 3 memories
    for i in 1..=3 {
        let memory = Memory {
            id: format!("mem-lg-{}", i),
            agent_id: "test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Memory {}", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("test"));
        vector_store.add(&format!("mem-lg-{}", i), &format!("Memory {}", i), None, meta).await.unwrap();
    }
    
    // Request k=1000 but only 3 exist
    let results = vector_store
        .search("Memory", 1000, Metadata::new(), Some("test"), None)
        .await
        .unwrap();
    
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_delete_nonexistent_embedding() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db, provider));
    
    // Should not error
    let result = vector_store.delete("nonexistent").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_search_with_empty_query() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    
    // Insert a memory
    let memory = Memory {
        id: "mem-empty-q".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Test memory".to_string(),
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
    meta.insert("agent_id".to_string(), serde_json::json!("test"));
    vector_store.add("mem-empty-q", "Test memory", None, meta).await.unwrap();
    
    // Empty query - the embedding will still be computed, behavior depends on implementation
    let results = vector_store
        .search("", 10, Metadata::new(), Some("test"), None)
        .await
        .unwrap();
    
    // Should still return results (empty string has an embedding)
    assert!(!results.is_empty());
}

// ============================================================================
// Event Handling Edge Cases  
// ============================================================================

#[tokio::test]
async fn test_event_with_special_characters() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    let event = MemoryEvent {
        id: "special-evt".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        event_type: "test".to_string(),
        content: "Special chars: 'quotes', \"double\", \n newline, \t tab, 日本語, 🎉".to_string(),
        metadata: None,
        created_at: Utc::now(),
        summarized_at: None,
    };
    
    db.insert_event(&event).await.unwrap();
    
    let events = db.list_unsummarized_events("test", None, None, 100).await.unwrap();
    assert_eq!(events.len(), 1);
    assert!(events[0].content.contains("日本語"));
    assert!(events[0].content.contains("🎉"));
}

#[tokio::test]
async fn test_event_with_very_long_content() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Create a very long string (100KB)
    let long_content = "a".repeat(100_000);
    
    let event = MemoryEvent {
        id: "long-evt".to_string(),
        agent_id: "test".to_string(),
        user_id: None,
        session_id: None,
        event_type: "test".to_string(),
        content: long_content.clone(),
        metadata: None,
        created_at: Utc::now(),
        summarized_at: None,
    };
    
    db.insert_event(&event).await.unwrap();
    
    let events = db.list_unsummarized_events("test", None, None, 100).await.unwrap();
    assert_eq!(events[0].content.len(), 100_000);
}

#[tokio::test]
async fn test_mark_summarized_empty_list() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Should not error with empty list
    let result = db.mark_events_summarized("test", &[]).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_mark_summarized_nonexistent_events() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Should not error with nonexistent IDs
    let result = db.mark_events_summarized(
        "test",
        &["fake-1".to_string(), "fake-2".to_string()]
    ).await;
    assert!(result.is_ok());
}

// ============================================================================
// Concurrency Edge Cases
// ============================================================================

#[tokio::test]
async fn test_concurrent_memory_insertions() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    
    // Spawn multiple concurrent inserts
    let mut handles = vec![];
    
    for i in 0..10 {
        let db_clone = db.clone();
        handles.push(tokio::spawn(async move {
            let memory = Memory {
                id: format!("concurrent-{}", i),
                agent_id: "test".to_string(),
                user_id: None,
                session_id: None,
                memory_type: "episodic".to_string(),
                text: format!("Concurrent memory {}", i),
                importance: 0.5,
                is_active: true,
                supersedes_id: None,
                source_event_ids: None,
                metadata: None,
                last_accessed_at: None,
                created_at: Utc::now(),
                expires_at: None,
            };
            db_clone.insert_memory(&memory).await
        }));
    }
    
    // Wait for all and check results
    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            success_count += 1;
        }
    }
    
    assert_eq!(success_count, 10);
}

#[tokio::test]
async fn test_concurrent_searches() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    
    // Insert some memories first
    for i in 0..5 {
        let memory = Memory {
            id: format!("search-mem-{}", i),
            agent_id: "test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Searchable memory {}", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("test"));
        vector_store.add(&format!("search-mem-{}", i), &format!("Searchable memory {}", i), None, meta).await.unwrap();
    }
    
    // Spawn concurrent searches
    let mut handles = vec![];
    
    for _ in 0..10 {
        let vs = vector_store.clone();
        handles.push(tokio::spawn(async move {
            vs.search("Searchable", 5, Metadata::new(), Some("test"), None).await
        }));
    }
    
    // All searches should succeed
    for handle in handles {
        let results = handle.await.unwrap().unwrap();
        assert!(!results.is_empty());
    }
}

