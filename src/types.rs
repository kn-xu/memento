use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Metadata = HashMap<String, serde_json::Value>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreRequest {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: Option<String>,
    pub text: Option<String>,
    pub content: Option<String>,
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    pub filters: Option<Metadata>,
}

fn default_k() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeRequest {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub query: Option<String>,
    pub memory_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreResponse {
    pub ok: bool,
    pub event_id: String,
    pub memory_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memory_id: String,
    pub text: String,
    pub score: f64,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub ok: bool,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeResponse {
    pub ok: bool,
    pub created: usize,
    pub updated: usize,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetResponse {
    pub ok: bool,
    pub deleted: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreToolArgs {
    pub text: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: Option<String>,
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchToolArgs {
    pub query: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub k: Option<usize>,
    pub filters: Option<Metadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeToolArgs {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetToolArgs {
    pub agent_id: String,
    pub user_id: Option<String>,
    pub memory_id: Option<String>,
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkSummarizedToolArgs {
    pub agent_id: String,
    pub event_ids: Vec<String>,
}
