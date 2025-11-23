use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub embedding_provider: EmbeddingProvider,
    pub openai_api_key: Option<String>,
    pub embedding_model: String,
    pub port: u16,
    pub host: String,
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

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("MEMENTO_DATABASE_URL")
                .or_else(|_| env::var("DATABASE_URL"))
                .unwrap_or_else(|_| "sqlite://./memento.db".to_string()),
            embedding_provider: env::var("MEMENTO_EMBEDDING_PROVIDER")
                .or_else(|_| env::var("EMBEDDING_PROVIDER"))
                .unwrap_or_else(|_| "local".to_string())
                .into(),
            openai_api_key: env::var("MEMENTO_OPENAI_API_KEY")
                .or_else(|_| env::var("OPENAI_API_KEY"))
                .ok(),
            embedding_model: env::var("MEMENTO_EMBEDDING_MODEL")
                .or_else(|_| env::var("EMBEDDING_MODEL"))
                .unwrap_or_else(|_| "Xenova/all-MiniLM-L6-v2".to_string()),
            port: env::var("MEMENTO_PORT")
                .or_else(|_| env::var("PORT"))
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .unwrap_or(8000),
            host: env::var("MEMENTO_HOST")
                .or_else(|_| env::var("HOST"))
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
        }
    }
}

