use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub vector_store: VectorStoreType,
    pub embedding_provider: EmbeddingProvider,
    pub openai_api_key: Option<String>,
    pub embedding_model: String,
    pub port: u16,
    pub host: String,
    pub mcp_transport: McpTransport,
    pub mcp_sse_path: String,
    pub summarizer_enabled: bool,
    pub summarizer_llm_provider: String,
    pub openai_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorStoreType {
    #[serde(rename = "sqlite-vss")]
    SqliteVss,
    #[serde(rename = "pgvector")]
    Pgvector,
    #[serde(rename = "chroma")]
    Chroma,
}

impl Default for VectorStoreType {
    fn default() -> Self {
        Self::SqliteVss
    }
}

impl From<String> for VectorStoreType {
    fn from(s: String) -> Self {
        match s.as_str() {
            "pgvector" => Self::Pgvector,
            "chroma" => Self::Chroma,
            _ => Self::SqliteVss,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "openai")]
    OpenAi,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        Self::Local
    }
}

impl From<String> for EmbeddingProvider {
    fn from(s: String) -> Self {
        match s.as_str() {
            "openai" => Self::OpenAi,
            _ => Self::Local,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpTransport {
    #[serde(rename = "stdio")]
    Stdio,
    #[serde(rename = "sse")]
    Sse,
}

impl Default for McpTransport {
    fn default() -> Self {
        Self::Stdio
    }
}

impl From<String> for McpTransport {
    fn from(s: String) -> Self {
        match s.as_str() {
            "sse" => Self::Sse,
            _ => Self::Stdio,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite://./memento.db".to_string()),
            vector_store: env::var("VECTOR_STORE")
                .unwrap_or_else(|_| "sqlite-vss".to_string())
                .into(),
            embedding_provider: env::var("EMBEDDING_PROVIDER")
                .unwrap_or_else(|_| "local".to_string())
                .into(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            embedding_model: env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "Xenova/all-MiniLM-L6-v2".to_string()),
            port: env::var("PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .unwrap_or(8000),
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            mcp_transport: env::var("MCP_TRANSPORT")
                .unwrap_or_else(|_| "stdio".to_string())
                .into(),
            mcp_sse_path: env::var("MCP_SSE_PATH")
                .unwrap_or_else(|_| "/mcp/sse".to_string()),
            summarizer_enabled: env::var("SUMMARIZER_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            summarizer_llm_provider: env::var("SUMMARIZER_LLM_PROVIDER")
                .unwrap_or_else(|_| "openai".to_string()),
            openai_model: env::var("OPENAI_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
        }
    }
}

