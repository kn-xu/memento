//! Tests for vector_store module

use memento::database::{DatabaseClient, Memory};
use memento::embeddings::DummyEmbeddingProvider;
use memento::types::Metadata;
use memento::vector_store::{VectorSearchResult, VectorStore};

/// Helper function to compute cosine similarity (re-implemented for tests).
/// Mirrors the production implementation in vector_store.rs.
/// 
/// # Edge Cases Handled
/// - Different vector lengths → 0.0
/// - Zero-magnitude vectors → 0.0
/// - NaN/Infinity values in either vector → 0.0
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        
        // Check for NaN/Infinity - if any value is non-finite, return 0.0
        if !ai.is_finite() || !bi.is_finite() {
            return 0.0;
        }
        
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let result = dot_product / (norm_a.sqrt() * norm_b.sqrt());
    
    // Final defensive check
    if result.is_finite() {
        result
    } else {
        0.0
    }
}

#[test]
fn test_cosine_similarity_identical_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 0.001);
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_different_lengths_returns_zero() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0);
}

#[test]
fn test_cosine_similarity_zero_vector_returns_zero() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0);
}

#[test]
fn test_cosine_similarity_normalized_vectors() {
    // Two normalized vectors at 45 degrees
    let a = vec![1.0, 0.0];
    let b = vec![0.707107, 0.707107]; // ~45 degrees
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 0.707107).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_nan_in_first_vector() {
    let a = vec![f32::NAN, 1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "NaN in first vector should return 0.0");
}

#[test]
fn test_cosine_similarity_nan_in_second_vector() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![f32::NAN, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "NaN in second vector should return 0.0");
}

#[test]
fn test_cosine_similarity_infinity_handling() {
    let a = vec![f32::INFINITY, 1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "Infinity in vector should return 0.0");
    
    let a = vec![f32::NEG_INFINITY, 1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "Negative infinity in vector should return 0.0");
}

#[test]
fn test_cosine_similarity_mixed_nan_and_valid() {
    // NaN at the end - should still detect it
    let a = vec![1.0, 0.0, f32::NAN];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0, "NaN anywhere in vector should return 0.0");
}

#[test]
fn test_vector_search_result_creation() {
    let result = VectorSearchResult {
        memory_id: "mem-123".to_string(),
        score: 0.95,
        metadata: Metadata::new(),
    };
    assert_eq!(result.memory_id, "mem-123");
    assert_eq!(result.score, 0.95);
}

#[tokio::test]
async fn test_vector_store_add_and_search() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = VectorStore::new(db.clone(), provider);

    // Insert a memory first
    let memory = Memory {
        id: "mem-vs-001".to_string(),
        agent_id: "test-agent".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "User loves Rust programming".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: chrono::Utc::now(),
    };
    db.insert_memory(&memory).await.unwrap();

    // Add embedding
    let mut metadata = Metadata::new();
    metadata.insert("agent_id".to_string(), serde_json::json!("test-agent"));
    vector_store
        .add(
            "mem-vs-001",
            "User loves Rust programming",
            None,
            metadata,
        )
        .await
        .unwrap();

    // Search
    let results = vector_store
        .search(
            "User loves Rust programming",
            5,
            Metadata::new(),
            Some("test-agent"),
            None,
        )
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].memory_id, "mem-vs-001");
    // Same text should have high similarity
    assert!(results[0].score > 0.9);
}

#[tokio::test]
async fn test_vector_store_delete() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = VectorStore::new(db.clone(), provider);

    // Insert a memory
    let memory = Memory {
        id: "mem-vs-del".to_string(),
        agent_id: "test-agent".to_string(),
        user_id: None,
        session_id: None,
        memory_type: "episodic".to_string(),
        text: "Temporary memory".to_string(),
        importance: 0.5,
        is_active: true,
        supersedes_id: None,
        source_event_ids: None,
        metadata: None,
        last_accessed_at: None,
        created_at: chrono::Utc::now(),
    };
    db.insert_memory(&memory).await.unwrap();
    vector_store
        .add("mem-vs-del", "Temporary memory", None, Metadata::new())
        .await
        .unwrap();

    // Delete embedding
    vector_store.delete("mem-vs-del").await.unwrap();

    // Search should not find it (embedding is NULL)
    let results = vector_store
        .search(
            "Temporary memory",
            5,
            Metadata::new(),
            Some("test-agent"),
            None,
        )
        .await
        .unwrap();

    // Should not find the deleted memory
    assert!(results.iter().all(|r| r.memory_id != "mem-vs-del"));
}

#[tokio::test]
async fn test_vector_store_search_with_filters() {
    let db = DatabaseClient::new("sqlite::memory:", 384).await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = VectorStore::new(db.clone(), provider);

    // Insert memories for different agents
    for (id, agent) in [("mem-a1", "agent-a"), ("mem-b1", "agent-b")] {
        let memory = Memory {
            id: id.to_string(),
            agent_id: agent.to_string(),
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
        };
        db.insert_memory(&memory).await.unwrap();

        let mut meta = Metadata::new();
        meta.insert("agent_id".to_string(), serde_json::json!(agent));
        vector_store
            .add(id, "Test memory", None, meta)
            .await
            .unwrap();
    }

    // Search with agent filter
    let results = vector_store
        .search("Test memory", 10, Metadata::new(), Some("agent-a"), None)
        .await
        .unwrap();

    // Should only find agent-a's memory
    assert!(results.iter().all(|r| r.memory_id == "mem-a1"));
}

