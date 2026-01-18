//! Type definitions for MCP tool arguments and shared data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Flexible key-value metadata attached to memories.
pub type Metadata = HashMap<String, serde_json::Value>;

/// Search result returned by memento.search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memory_id: String,
    pub text: String,
    pub score: f64,
    pub metadata: Metadata,
}

/// Arguments for the memento.store tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreToolArgs {
    pub text: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: Option<String>,
    pub metadata: Option<Metadata>,
}

/// Arguments for the memento.search tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchToolArgs {
    pub query: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub k: Option<usize>,
    pub filters: Option<Metadata>,
}

/// Arguments for the memento.summarize tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeToolArgs {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub limit: Option<usize>,
}

/// Arguments for the memento.forget tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetToolArgs {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub memory_id: Option<String>,
    pub query: Option<String>,
}

/// Arguments for the memento.mark_summarized tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkSummarizedToolArgs {
    pub agent_id: String,
    pub event_ids: Vec<String>,
}
