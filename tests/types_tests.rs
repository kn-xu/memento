//! Tests for types module

use memento::types::*;

#[test]
fn test_store_request_serialization() {
    let request = StoreRequest {
        agent_id: "cursor".to_string(),
        user_id: Some("user-123".to_string()),
        session_id: None,
        event_type: Some("user_msg".to_string()),
        text: Some("Hello, world!".to_string()),
        content: None,
        metadata: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    let deserialized: StoreRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.agent_id, "cursor");
    assert_eq!(deserialized.user_id, Some("user-123".to_string()));
    assert_eq!(deserialized.text, Some("Hello, world!".to_string()));
}

#[test]
fn test_store_request_with_metadata() {
    let mut metadata = Metadata::new();
    metadata.insert(
        "tags".to_string(),
        serde_json::json!(["preference", "coding"]),
    );

    let request = StoreRequest {
        agent_id: "cursor".to_string(),
        user_id: None,
        session_id: None,
        event_type: None,
        text: Some("I prefer Rust".to_string()),
        content: None,
        metadata: Some(metadata),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("tags"));
    assert!(json.contains("preference"));
}

#[test]
fn test_search_request_default_k() {
    let json = r#"{"agent_id": "cursor", "query": "test query"}"#;
    let request: SearchRequest = serde_json::from_str(json).unwrap();

    assert_eq!(request.k, 5); // default value
}

#[test]
fn test_search_request_custom_k() {
    let json = r#"{"agent_id": "cursor", "query": "test query", "k": 10}"#;
    let request: SearchRequest = serde_json::from_str(json).unwrap();

    assert_eq!(request.k, 10);
}

#[test]
fn test_search_response_serialization() {
    let mut metadata = Metadata::new();
    metadata.insert("memory_type".to_string(), serde_json::json!("episodic"));

    let response = SearchResponse {
        ok: true,
        results: vec![
            SearchResult {
                memory_id: "mem-001".to_string(),
                text: "Test memory".to_string(),
                score: 0.95,
                metadata: metadata.clone(),
            },
            SearchResult {
                memory_id: "mem-002".to_string(),
                text: "Another memory".to_string(),
                score: 0.85,
                metadata: Metadata::new(),
            },
        ],
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: SearchResponse = serde_json::from_str(&json).unwrap();

    assert!(deserialized.ok);
    assert_eq!(deserialized.results.len(), 2);
    assert_eq!(deserialized.results[0].score, 0.95);
}

#[test]
fn test_store_response_serialization() {
    let response = StoreResponse {
        ok: true,
        event_id: "evt-123".to_string(),
        memory_id: Some("mem-456".to_string()),
    };

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: StoreResponse = serde_json::from_str(&json).unwrap();

    assert!(deserialized.ok);
    assert_eq!(deserialized.event_id, "evt-123");
    assert_eq!(deserialized.memory_id, Some("mem-456".to_string()));
}

#[test]
fn test_forget_request_with_memory_id() {
    let request = ForgetRequest {
        agent_id: "cursor".to_string(),
        user_id: None,
        query: None,
        memory_id: Some("mem-to-forget".to_string()),
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("mem-to-forget"));
}

#[test]
fn test_forget_request_with_query() {
    let request = ForgetRequest {
        agent_id: "cursor".to_string(),
        user_id: None,
        query: Some("phone number".to_string()),
        memory_id: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("phone number"));
}

#[test]
fn test_summarize_request_serialization() {
    let request = SummarizeRequest {
        agent_id: "cursor".to_string(),
        user_id: Some("user-1".to_string()),
        session_id: Some("session-1".to_string()),
    };

    let json = serde_json::to_string(&request).unwrap();
    let deserialized: SummarizeRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.agent_id, "cursor");
    assert_eq!(deserialized.session_id, Some("session-1".to_string()));
}

#[test]
fn test_health_response() {
    let response = HealthResponse {
        status: "ok".to_string(),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("ok"));
}

#[test]
fn test_error_response() {
    let response = ErrorResponse {
        error: "Something went wrong".to_string(),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("Something went wrong"));
}

#[test]
fn test_store_tool_args_deserialization() {
    let json = r#"{
        "text": "User prefers TypeScript",
        "agent_id": "cursor",
        "event_type": "preference"
    }"#;

    let args: StoreToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.text, "User prefers TypeScript");
    assert_eq!(args.agent_id, "cursor");
    assert_eq!(args.event_type, Some("preference".to_string()));
    assert!(args.user_id.is_none());
}

#[test]
fn test_search_tool_args_deserialization() {
    let json = r#"{
        "query": "coding preferences",
        "agent_id": "cursor",
        "k": 10
    }"#;

    let args: SearchToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.query, "coding preferences");
    assert_eq!(args.k, Some(10));
}

#[test]
fn test_metadata_type() {
    let mut metadata: Metadata = Metadata::new();
    metadata.insert("key".to_string(), serde_json::json!("value"));
    metadata.insert("number".to_string(), serde_json::json!(42));
    metadata.insert("array".to_string(), serde_json::json!(["a", "b", "c"]));

    assert_eq!(metadata.get("key").unwrap(), &serde_json::json!("value"));
    assert_eq!(metadata.get("number").unwrap(), &serde_json::json!(42));
}

