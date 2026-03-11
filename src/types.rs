//! Type definitions for MCP tool arguments and shared data structures.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Flexible key-value metadata attached to memories.
pub type Metadata = HashMap<String, serde_json::Value>;

// =============================================================================
// Salience types
// =============================================================================

/// Overall emotional salience score attached to a memory at ingestion time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SalienceScore {
    /// 0.0–1.0  overall salience
    pub value: f32,
    /// -1.0 (negative) – 1.0 (positive)
    pub valence: f32,
    /// 0.0–1.0  scorer confidence
    pub confidence: f32,
    pub signals: SalienceSignals,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SalienceSignals {
    pub punctuation_score: f32,
    pub length_deviation: f32,
    pub correction_detected: bool,
    pub urgency_detected: bool,
    pub retry_count: u32,
    pub explicit_emotion: Option<EmotionTag>,
    /// None when onnx feature is disabled
    pub onnx_emotion: Option<OnnxEmotion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionTag {
    Frustration,
    Excitement,
    Confusion,
    Satisfaction,
    Urgency,
    Neutral,
}

impl EmotionTag {
    pub fn salience_weight(&self) -> f32 {
        match self {
            EmotionTag::Frustration => 0.80,
            EmotionTag::Excitement  => 0.75,
            EmotionTag::Urgency     => 0.85,
            EmotionTag::Confusion   => 0.60,
            EmotionTag::Satisfaction => 0.50,
            EmotionTag::Neutral     => 0.10,
        }
    }

    pub fn valence(&self) -> f32 {
        match self {
            EmotionTag::Frustration  => -0.70,
            EmotionTag::Excitement   =>  0.80,
            EmotionTag::Urgency      => -0.40,
            EmotionTag::Confusion    => -0.30,
            EmotionTag::Satisfaction =>  0.70,
            EmotionTag::Neutral      =>  0.00,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxEmotion {
    pub label: String,
    pub confidence: f32,
    pub salience_weight: f32,
}

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
    /// Optional importance override (0.0-1.0). If not provided, importance is calculated automatically.
    pub importance: Option<f64>,
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

/// Supported database types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatabaseType {
    Sqlite,
    #[serde(alias = "postgres")]
    Postgresql,
}

impl std::fmt::Display for DatabaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatabaseType::Sqlite => write!(f, "sqlite"),
            DatabaseType::Postgresql => write!(f, "postgresql"),
        }
    }
}

/// Supported embedding providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderType {
    Local,
    #[serde(alias = "openai")]
    Openai,
}

impl std::fmt::Display for EmbeddingProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingProviderType::Local => write!(f, "local"),
            EmbeddingProviderType::Openai => write!(f, "openai"),
        }
    }
}

/// Arguments for the memento.configure tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureToolArgs {
    /// Database type: "sqlite" or "postgresql"
    pub database_type: Option<DatabaseType>,
    /// Database URL or path
    pub database_url: Option<String>,
    /// Embedding provider: "local" or "openai"
    pub embedding_provider: Option<EmbeddingProviderType>,
    /// Embedding model name
    pub embedding_model: Option<String>,
    /// Embedding dimension
    pub embedding_dim: Option<usize>,
}
