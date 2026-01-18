//! Tests for MCP types module

use memento::types::*;

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
fn test_store_tool_args_with_metadata() {
    let json = r#"{
        "text": "I prefer Rust",
        "agent_id": "cursor",
        "metadata": {"tags": ["preference", "coding"]}
    }"#;

    let args: StoreToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.text, "I prefer Rust");
    assert!(args.metadata.is_some());
    let meta = args.metadata.unwrap();
    assert!(meta.contains_key("tags"));
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
fn test_search_tool_args_default_k() {
    let json = r#"{
        "query": "test query",
        "agent_id": "cursor"
    }"#;

    let args: SearchToolArgs = serde_json::from_str(json).unwrap();
    assert!(args.k.is_none()); // k is optional, defaults handled in handler
}

#[test]
fn test_summarize_tool_args_deserialization() {
    let json = r#"{
        "agent_id": "cursor",
        "user_id": "user-1",
        "limit": 100
    }"#;

    let args: SummarizeToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.agent_id, "cursor");
    assert_eq!(args.user_id, Some("user-1".to_string()));
    assert_eq!(args.limit, Some(100));
}

#[test]
fn test_forget_tool_args_with_memory_id() {
    let json = r#"{
        "agent_id": "cursor",
        "memory_id": "mem-to-forget"
    }"#;

    let args: ForgetToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.memory_id, Some("mem-to-forget".to_string()));
    assert!(args.query.is_none());
}

#[test]
fn test_forget_tool_args_with_query() {
    let json = r#"{
        "agent_id": "cursor",
        "query": "phone number"
    }"#;

    let args: ForgetToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.query, Some("phone number".to_string()));
    assert!(args.memory_id.is_none());
}

#[test]
fn test_mark_summarized_tool_args() {
    let json = r#"{
        "agent_id": "cursor",
        "event_ids": ["evt-1", "evt-2", "evt-3"]
    }"#;

    let args: MarkSummarizedToolArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.agent_id, "cursor");
    assert_eq!(args.event_ids.len(), 3);
    assert_eq!(args.event_ids[0], "evt-1");
}

#[test]
fn test_search_result_serialization() {
    let mut metadata = Metadata::new();
    metadata.insert("memory_type".to_string(), serde_json::json!("episodic"));

    let result = SearchResult {
        memory_id: "mem-001".to_string(),
        text: "Test memory".to_string(),
        score: 0.95,
        metadata,
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.memory_id, "mem-001");
    assert_eq!(deserialized.score, 0.95);
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
