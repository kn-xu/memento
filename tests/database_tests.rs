//! Tests for database module

use memento::database::{DatabaseClient, Memory, MemoryEvent};

#[test]
fn test_normalize_sqlite_url_plain_path() {
    let result = DatabaseClient::normalize_sqlite_url("./test.db");
    assert_eq!(result, "sqlite://./test.db");
}

#[test]
fn test_normalize_sqlite_url_already_prefixed() {
    let result = DatabaseClient::normalize_sqlite_url("sqlite://./test.db");
    assert_eq!(result, "sqlite://./test.db");
}

#[test]
fn test_normalize_sqlite_url_memory() {
    let result = DatabaseClient::normalize_sqlite_url(":memory:");
    assert_eq!(result, "sqlite::memory:");
}

#[test]
fn test_normalize_sqlite_url_sqlite_memory() {
    let result = DatabaseClient::normalize_sqlite_url("sqlite::memory:");
    assert_eq!(result, "sqlite::memory:");
}

#[test]
fn test_normalize_sqlite_url_with_query_params() {
    let result = DatabaseClient::normalize_sqlite_url("./test.db?mode=rwc");
    assert_eq!(result, "sqlite://./test.db");
}

#[test]
fn test_normalize_sqlite_url_absolute_path() {
    let result = DatabaseClient::normalize_sqlite_url("/var/data/memento.db");
    assert_eq!(result, "sqlite:///var/data/memento.db");
}

#[test]
fn test_normalize_sqlite_url_double_colon_format() {
    let result = DatabaseClient::normalize_sqlite_url("sqlite::./test.db");
    assert_eq!(result, "sqlite://./test.db");
}

#[tokio::test]
async fn test_sqlite_database_creation() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    assert!(matches!(db, DatabaseClient::Sqlite(_)));
}

#[tokio::test]
async fn test_insert_and_get_event() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let event = MemoryEvent {
        id: "evt-001".to_string(),
        agent_id: "test-agent".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: Some("session-1".to_string()),
        event_type: "user_msg".to_string(),
        content: "Hello, world!".to_string(),
        metadata: Some(r#"{"key": "value"}"#.to_string()),
        created_at: chrono::Utc::now(),
        summarized_at: None,
    };

    db.insert_event(&event).await.unwrap();

    // Verify by listing unsummarized events
    let events = db
        .list_unsummarized_events("test-agent", Some("user-1"), None, 10)
        .await
        .unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].id, "evt-001");
    assert_eq!(events[0].content, "Hello, world!");
}

#[tokio::test]
async fn test_insert_and_get_memory() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-001".to_string(),
        agent_id: "test-agent".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "User prefers dark mode".to_string(),
        importance: 0.8,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: chrono::Utc::now(),
        salience: None,
    };

    db.insert_memory(&memory).await.unwrap();

    let retrieved = db.get_memory("mem-001").await.unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.text, "User prefers dark mode");
    assert_eq!(retrieved.importance, 0.8);
    assert!(retrieved.is_active);
}

#[tokio::test]
async fn test_soft_delete_memory() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-002".to_string(),
        agent_id: "test-agent".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "semantic".to_string(),
        text: "Some memory".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: chrono::Utc::now(),
        salience: None,
    };

    db.insert_memory(&memory).await.unwrap();
    db.soft_delete_memory("mem-002").await.unwrap();

    let retrieved = db.get_memory("mem-002").await.unwrap().unwrap();
    assert!(!retrieved.is_active);
}

#[tokio::test]
async fn test_mark_events_summarized() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    for i in 1..=3 {
        let event = MemoryEvent {
            id: format!("evt-{}", i),
            agent_id: "test-agent".to_string(),
            user_id: None,
            session_id: None,
            event_type: "user_msg".to_string(),
            content: format!("Event {}", i),
            metadata: None,
            created_at: chrono::Utc::now(),
            summarized_at: None,
        };
        db.insert_event(&event).await.unwrap();
    }

    // Initially all unsummarized
    let unsummarized = db
        .list_unsummarized_events("test-agent", None, None, 10)
        .await
        .unwrap();
    assert_eq!(unsummarized.len(), 3);

    // Mark first two as summarized
    db.mark_events_summarized("test-agent", &["evt-1".to_string(), "evt-2".to_string()])
        .await
        .unwrap();

    // Only one should remain unsummarized
    let unsummarized = db
        .list_unsummarized_events("test-agent", None, None, 10)
        .await
        .unwrap();
    assert_eq!(unsummarized.len(), 1);
    assert_eq!(unsummarized[0].id, "evt-3");
}

#[tokio::test]
async fn test_get_memories_by_ids() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    for i in 1..=3 {
        let memory = Memory {
            id: format!("mem-{}", i),
            agent_id: "test-agent".to_string(),
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
            created_at: chrono::Utc::now(),
            salience: None,
        };
        db.insert_memory(&memory).await.unwrap();
    }

    let ids = vec!["mem-1".to_string(), "mem-3".to_string()];
    let memories = db.get_memories_by_ids(&ids).await.unwrap();

    assert_eq!(memories.len(), 2);
    assert!(memories.contains_key("mem-1"));
    assert!(memories.contains_key("mem-3"));
    assert!(!memories.contains_key("mem-2"));
}

#[tokio::test]
async fn test_get_nonexistent_memory() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let result = db.get_memory("nonexistent").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_update_memory_access() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-access".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };

    db.insert_memory(&memory).await.unwrap();

    // Initially no last_accessed_at
    let retrieved = db.get_memory("mem-access").await.unwrap().unwrap();
    assert!(retrieved.last_accessed_at.is_none());

    // Update access
    db.update_memory_access("mem-access").await.unwrap();

    let retrieved = db.get_memory("mem-access").await.unwrap().unwrap();
    assert!(retrieved.last_accessed_at.is_some());
}

// =============================================================================
// Boost Importance Edge Cases
// =============================================================================

#[tokio::test]
async fn test_boost_importance_normal() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-boost".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Normal boost should increase importance
    db.boost_importance("mem-boost", 0.1, 1.0).await.unwrap();

    let retrieved = db.get_memory("mem-boost").await.unwrap().unwrap();
    assert!(retrieved.importance > 0.5, "Importance should have increased");
    assert!(retrieved.importance < 0.7, "Boost should be modest with diminishing returns");
}

#[tokio::test]
async fn test_boost_importance_zero_max_importance() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-boost-zero".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Zero max_importance should be a no-op (prevents division by zero)
    db.boost_importance("mem-boost-zero", 0.1, 0.0).await.unwrap();

    let retrieved = db.get_memory("mem-boost-zero").await.unwrap().unwrap();
    assert_eq!(retrieved.importance, 0.5, "Importance should be unchanged");
}

#[tokio::test]
async fn test_boost_importance_negative_max_importance() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-boost-neg".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Negative max_importance should be a no-op
    db.boost_importance("mem-boost-neg", 0.1, -1.0).await.unwrap();

    let retrieved = db.get_memory("mem-boost-neg").await.unwrap().unwrap();
    assert_eq!(retrieved.importance, 0.5, "Importance should be unchanged");
}

#[tokio::test]
async fn test_boost_importance_zero_boost_amount() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-boost-zero-amt".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Zero boost_amount should be a no-op
    db.boost_importance("mem-boost-zero-amt", 0.0, 1.0).await.unwrap();

    let retrieved = db.get_memory("mem-boost-zero-amt").await.unwrap().unwrap();
    assert_eq!(retrieved.importance, 0.5, "Importance should be unchanged");
}

#[tokio::test]
async fn test_boost_importance_negative_boost_amount() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();

    let memory = Memory {
        id: "mem-boost-neg-amt".to_string(),
        agent_id: "test-agent".to_string(),
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
        created_at: chrono::Utc::now(),
        salience: None,
    };
    db.insert_memory(&memory).await.unwrap();

    // Negative boost_amount should be a no-op
    db.boost_importance("mem-boost-neg-amt", -0.1, 1.0).await.unwrap();

    let retrieved = db.get_memory("mem-boost-neg-amt").await.unwrap().unwrap();
    assert_eq!(retrieved.importance, 0.5, "Importance should be unchanged");
}
