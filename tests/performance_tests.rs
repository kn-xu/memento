//! Performance tests for tool call latency and throughput
//!
//! These tests measure the performance characteristics of the memento system
//! including latency, throughput, and scalability.

use memento::database::{DatabaseClient, Memory, MemoryEvent};
use memento::embeddings::DummyEmbeddingProvider;
use memento::types::Metadata;
use memento::vector_store::VectorStore;
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Helper to setup a complete test environment
async fn setup_perf_env() -> (DatabaseClient, Arc<VectorStore>) {
    let db = DatabaseClient::new("sqlite::memory:").await.unwrap();
    let provider = Box::new(DummyEmbeddingProvider::new(None, Some(384)));
    let vector_store = Arc::new(VectorStore::new(db.clone(), provider));
    (db, vector_store)
}

/// Performance result for reporting
#[derive(Debug)]
struct PerfResult {
    operation: String,
    iterations: u32,
    total_time: Duration,
    avg_latency: Duration,
    min_latency: Duration,
    max_latency: Duration,
    ops_per_sec: f64,
}

impl PerfResult {
    fn report(&self) {
        println!("\n=== {} Performance ===", self.operation);
        println!("  Iterations:    {}", self.iterations);
        println!("  Total time:    {:?}", self.total_time);
        println!("  Avg latency:   {:?}", self.avg_latency);
        println!("  Min latency:   {:?}", self.min_latency);
        println!("  Max latency:   {:?}", self.max_latency);
        println!("  Throughput:    {:.2} ops/sec", self.ops_per_sec);
    }
}

// ============================================================================
// Store Performance Tests
// ============================================================================

#[tokio::test]
async fn test_perf_store_latency() {
    let (db, vector_store) = setup_perf_env().await;
    let iterations = 100;
    let mut latencies = Vec::with_capacity(iterations);

    let start_total = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();

        // Create event
        let event_id = Uuid::new_v4().to_string();
        let event = MemoryEvent {
            id: event_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: Some("user".to_string()),
            session_id: None,
            event_type: "test".to_string(),
            content: format!("Performance test memory {}", i),
            metadata: None,
            created_at: Utc::now(),
            summarized_at: None,
        };
        db.insert_event(&event).await.unwrap();

        // Create memory
        let memory_id = Uuid::new_v4().to_string();
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: Some("user".to_string()),
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Performance test memory {}", i),
            importance: 0.5,
            is_active: true,
            supersedes_id: None,
            source_event_ids: Some(format!(r#"["{}"]"#, event_id)),
            metadata: None,
            last_accessed_at: None,
            created_at: Utc::now(),
            expires_at: None,
        };
        db.insert_memory(&memory).await.unwrap();

        // Add embedding
        let mut meta = Metadata::new();
        meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
        vector_store
            .add(&memory_id, &format!("Performance test memory {}", i), None, meta)
            .await
            .unwrap();

        latencies.push(start.elapsed());
    }

    let total_time = start_total.elapsed();
    let result = PerfResult {
        operation: "Store (event + memory + embedding)".to_string(),
        iterations: iterations as u32,
        total_time,
        avg_latency: total_time / iterations as u32,
        min_latency: *latencies.iter().min().unwrap(),
        max_latency: *latencies.iter().max().unwrap(),
        ops_per_sec: iterations as f64 / total_time.as_secs_f64(),
    };

    result.report();

    // Assert performance requirements
    assert!(result.avg_latency < Duration::from_millis(50), "Store avg latency too high: {:?}", result.avg_latency);
}

// ============================================================================
// Search Performance Tests
// ============================================================================

#[tokio::test]
async fn test_perf_search_latency() {
    let (db, vector_store) = setup_perf_env().await;

    // Pre-populate with 500 memories
    for i in 0..500 {
        let memory_id = format!("search-perf-{}", i);
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: Some("user".to_string()),
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Searchable memory about topic {} with unique content", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
        meta.insert("user_id".to_string(), serde_json::json!("user"));
        vector_store
            .add(&memory_id, &format!("Searchable memory about topic {} with unique content", i), None, meta)
            .await
            .unwrap();
    }

    let iterations = 100;
    let queries = vec![
        "topic unique content",
        "searchable memory",
        "about topic",
        "unique searchable",
        "memory content topic",
    ];
    let mut latencies = Vec::with_capacity(iterations);

    let start_total = Instant::now();

    for i in 0..iterations {
        let query = &queries[i % queries.len()];
        let start = Instant::now();

        let results = vector_store
            .search(query, 10, Metadata::new(), Some("perf-test"), Some("user"))
            .await
            .unwrap();

        latencies.push(start.elapsed());
        assert!(!results.is_empty());
    }

    let total_time = start_total.elapsed();
    let result = PerfResult {
        operation: "Search (k=10, 500 memories)".to_string(),
        iterations: iterations as u32,
        total_time,
        avg_latency: total_time / iterations as u32,
        min_latency: *latencies.iter().min().unwrap(),
        max_latency: *latencies.iter().max().unwrap(),
        ops_per_sec: iterations as f64 / total_time.as_secs_f64(),
    };

    result.report();

    // Assert performance requirements
    assert!(result.avg_latency < Duration::from_millis(20), "Search avg latency too high: {:?}", result.avg_latency);
}

#[tokio::test]
async fn test_perf_search_scaling() {
    println!("\n=== Search Scaling Test ===");
    
    let sizes = [100, 250, 500, 1000];
    
    for size in sizes {
        let (db, vector_store) = setup_perf_env().await;

        // Pre-populate
        for i in 0..size {
            let memory_id = format!("scale-{}", i);
            let memory = Memory {
                id: memory_id.clone(),
                agent_id: "perf-test".to_string(),
                user_id: None,
                session_id: None,
                memory_type: "episodic".to_string(),
                text: format!("Memory number {} for scaling test", i),
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
            meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
            vector_store
                .add(&memory_id, &format!("Memory number {} for scaling test", i), None, meta)
                .await
                .unwrap();
        }

        // Benchmark
        let iterations = 50;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = vector_store
                .search("memory scaling test", 10, Metadata::new(), Some("perf-test"), None)
                .await
                .unwrap();
        }

        let elapsed = start.elapsed();
        let avg_latency = elapsed / iterations as u32;
        println!("  {} memories: avg {:?} per search", size, avg_latency);
        
        // Scaling should be sub-linear (SQLite is O(n) scan, but should still be fast)
        assert!(avg_latency < Duration::from_millis(100), "Search too slow at {} memories", size);
    }
}

// ============================================================================
// Concurrent Performance Tests
// ============================================================================

#[tokio::test]
async fn test_perf_concurrent_stores() {
    let (db, vector_store) = setup_perf_env().await;
    let concurrent_tasks = 20;
    let stores_per_task = 10;

    let start = Instant::now();

    let mut handles = vec![];
    for task_id in 0..concurrent_tasks {
        let db_clone = db.clone();
        let vs_clone = vector_store.clone();
        
        handles.push(tokio::spawn(async move {
            for i in 0..stores_per_task {
                let event_id = Uuid::new_v4().to_string();
                let event = MemoryEvent {
                    id: event_id.clone(),
                    agent_id: format!("task-{}", task_id),
                    user_id: None,
                    session_id: None,
                    event_type: "test".to_string(),
                    content: format!("Concurrent store {} from task {}", i, task_id),
                    metadata: None,
                    created_at: Utc::now(),
                    summarized_at: None,
                };
                db_clone.insert_event(&event).await.unwrap();

                let memory_id = Uuid::new_v4().to_string();
                let memory = Memory {
                    id: memory_id.clone(),
                    agent_id: format!("task-{}", task_id),
                    user_id: None,
                    session_id: None,
                    memory_type: "episodic".to_string(),
                    text: format!("Concurrent store {} from task {}", i, task_id),
                    importance: 0.5,
                    is_active: true,
                    supersedes_id: None,
                    source_event_ids: None,
                    metadata: None,
                    last_accessed_at: None,
                    created_at: Utc::now(),
                    expires_at: None,
                };
                db_clone.insert_memory(&memory).await.unwrap();

                let mut meta = Metadata::new();
                meta.insert("agent_id".to_string(), serde_json::json!(format!("task-{}", task_id)));
                vs_clone
                    .add(&memory_id, &format!("Concurrent store {} from task {}", i, task_id), None, meta)
                    .await
                    .unwrap();
            }
        }));
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap();
    }

    let total_time = start.elapsed();
    let total_ops = concurrent_tasks * stores_per_task;
    let ops_per_sec = total_ops as f64 / total_time.as_secs_f64();

    println!("\n=== Concurrent Stores Performance ===");
    println!("  Tasks:         {}", concurrent_tasks);
    println!("  Stores/task:   {}", stores_per_task);
    println!("  Total stores:  {}", total_ops);
    println!("  Total time:    {:?}", total_time);
    println!("  Throughput:    {:.2} ops/sec", ops_per_sec);

    assert!(ops_per_sec > 50.0, "Concurrent store throughput too low: {:.2}", ops_per_sec);
}

#[tokio::test]
async fn test_perf_concurrent_searches() {
    let (db, vector_store) = setup_perf_env().await;

    // Pre-populate
    for i in 0..300 {
        let memory_id = format!("conc-search-{}", i);
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Concurrent search test memory {}", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
        vector_store
            .add(&memory_id, &format!("Concurrent search test memory {}", i), None, meta)
            .await
            .unwrap();
    }

    let concurrent_tasks = 20;
    let searches_per_task = 20;

    let start = Instant::now();

    let mut handles = vec![];
    for _ in 0..concurrent_tasks {
        let vs_clone = vector_store.clone();
        
        handles.push(tokio::spawn(async move {
            for _ in 0..searches_per_task {
                let results = vs_clone
                    .search("concurrent search test memory", 5, Metadata::new(), Some("perf-test"), None)
                    .await
                    .unwrap();
                assert!(!results.is_empty());
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let total_time = start.elapsed();
    let total_ops = concurrent_tasks * searches_per_task;
    let ops_per_sec = total_ops as f64 / total_time.as_secs_f64();

    println!("\n=== Concurrent Searches Performance ===");
    println!("  Tasks:          {}", concurrent_tasks);
    println!("  Searches/task:  {}", searches_per_task);
    println!("  Total searches: {}", total_ops);
    println!("  Total time:     {:?}", total_time);
    println!("  Throughput:     {:.2} ops/sec", ops_per_sec);

    assert!(ops_per_sec > 100.0, "Concurrent search throughput too low: {:.2}", ops_per_sec);
}

// ============================================================================
// Mixed Workload Performance Tests
// ============================================================================

#[tokio::test]
async fn test_perf_mixed_workload() {
    let (db, vector_store) = setup_perf_env().await;

    // Pre-populate with some base data
    for i in 0..100 {
        let memory_id = format!("base-{}", i);
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "mixed-test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Base memory for mixed workload test {}", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("mixed-test"));
        vector_store
            .add(&memory_id, &format!("Base memory for mixed workload test {}", i), None, meta)
            .await
            .unwrap();
    }

    // Mixed workload: 70% reads, 30% writes
    let iterations = 100;
    let mut read_latencies = vec![];
    let mut write_latencies = vec![];

    let start = Instant::now();

    for i in 0..iterations {
        if i % 10 < 7 {
            // Read (search)
            let read_start = Instant::now();
            let _ = vector_store
                .search("mixed workload test", 5, Metadata::new(), Some("mixed-test"), None)
                .await
                .unwrap();
            read_latencies.push(read_start.elapsed());
        } else {
            // Write (store)
            let write_start = Instant::now();
            
            let memory_id = format!("mixed-new-{}", i);
            let memory = Memory {
                id: memory_id.clone(),
                agent_id: "mixed-test".to_string(),
                user_id: None,
                session_id: None,
                memory_type: "episodic".to_string(),
                text: format!("New memory from mixed workload {}", i),
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
            meta.insert("agent_id".to_string(), serde_json::json!("mixed-test"));
            vector_store
                .add(&memory_id, &format!("New memory from mixed workload {}", i), None, meta)
                .await
                .unwrap();
            
            write_latencies.push(write_start.elapsed());
        }
    }

    let total_time = start.elapsed();

    println!("\n=== Mixed Workload Performance (70% read, 30% write) ===");
    println!("  Total operations: {}", iterations);
    println!("  Total time:       {:?}", total_time);
    println!("  Reads:            {}", read_latencies.len());
    println!("  Writes:           {}", write_latencies.len());
    
    if !read_latencies.is_empty() {
        let avg_read: Duration = read_latencies.iter().sum::<Duration>() / read_latencies.len() as u32;
        println!("  Avg read latency:  {:?}", avg_read);
    }
    
    if !write_latencies.is_empty() {
        let avg_write: Duration = write_latencies.iter().sum::<Duration>() / write_latencies.len() as u32;
        println!("  Avg write latency: {:?}", avg_write);
    }

    let ops_per_sec = iterations as f64 / total_time.as_secs_f64();
    println!("  Throughput:       {:.2} ops/sec", ops_per_sec);

    assert!(ops_per_sec > 30.0, "Mixed workload throughput too low");
}

// ============================================================================
// Memory Overhead Tests
// ============================================================================

#[tokio::test]
async fn test_perf_large_content_handling() {
    let (db, vector_store) = setup_perf_env().await;

    // Test with increasingly large content
    let sizes = [100, 1_000, 10_000, 50_000];

    println!("\n=== Large Content Handling ===");

    for size in sizes {
        let content = "a".repeat(size);
        
        let start = Instant::now();
        
        let memory_id = format!("large-{}", size);
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: content.clone(),
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
        meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
        vector_store.add(&memory_id, &content, None, meta).await.unwrap();

        let store_time = start.elapsed();
        
        // Search
        let search_start = Instant::now();
        let _ = vector_store
            .search(&content[..100.min(size)], 5, Metadata::new(), Some("perf-test"), None)
            .await
            .unwrap();
        let search_time = search_start.elapsed();

        println!("  {} bytes: store {:?}, search {:?}", size, store_time, search_time);
    }
}

// ============================================================================
// Latency Distribution Test
// ============================================================================

#[tokio::test]
async fn test_perf_latency_distribution() {
    let (db, vector_store) = setup_perf_env().await;

    // Pre-populate
    for i in 0..200 {
        let memory_id = format!("dist-{}", i);
        let memory = Memory {
            id: memory_id.clone(),
            agent_id: "perf-test".to_string(),
            user_id: None,
            session_id: None,
            memory_type: "episodic".to_string(),
            text: format!("Distribution test memory content {}", i),
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
        meta.insert("agent_id".to_string(), serde_json::json!("perf-test"));
        vector_store
            .add(&memory_id, &format!("Distribution test memory content {}", i), None, meta)
            .await
            .unwrap();
    }

    let iterations = 100;
    let mut latencies: Vec<Duration> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = vector_store
            .search("distribution test memory", 5, Metadata::new(), Some("perf-test"), None)
            .await
            .unwrap();
        latencies.push(start.elapsed());
    }

    latencies.sort();

    let p50 = latencies[iterations * 50 / 100];
    let p90 = latencies[iterations * 90 / 100];
    let p99 = latencies[iterations * 99 / 100];

    println!("\n=== Search Latency Distribution (200 memories) ===");
    println!("  p50:  {:?}", p50);
    println!("  p90:  {:?}", p90);
    println!("  p99:  {:?}", p99);

    // Assert percentile requirements (relaxed for CI environments)
    assert!(p50 < Duration::from_millis(50), "p50 latency too high: {:?}", p50);
    assert!(p90 < Duration::from_millis(100), "p90 latency too high: {:?}", p90);
    assert!(p99 < Duration::from_millis(200), "p99 latency too high: {:?}", p99);
}

