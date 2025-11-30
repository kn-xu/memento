//! Tests for config module

use memento::config::{Config, EmbeddingProvider};

#[test]
fn test_embedding_provider_default() {
    let provider = EmbeddingProvider::default();
    assert!(matches!(provider, EmbeddingProvider::Local));
}

#[test]
fn test_embedding_provider_from_string_openai() {
    let provider: EmbeddingProvider = "openai".to_string().into();
    assert!(matches!(provider, EmbeddingProvider::OpenAi));
}

#[test]
fn test_embedding_provider_from_string_local() {
    let provider: EmbeddingProvider = "local".to_string().into();
    assert!(matches!(provider, EmbeddingProvider::Local));
}

#[test]
fn test_embedding_provider_from_string_unknown_defaults_to_local() {
    let provider: EmbeddingProvider = "unknown".to_string().into();
    assert!(matches!(provider, EmbeddingProvider::Local));
}

#[test]
fn test_config_serialization() {
    let config = Config {
        database_url: "sqlite://test.db".to_string(),
        embedding_provider: EmbeddingProvider::Local,
        openai_api_key: None,
        embedding_model: "test-model".to_string(),
        embedding_dim: Some(384),
        port: 8080,
        host: "127.0.0.1".to_string(),
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: Config = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.database_url, "sqlite://test.db");
    assert_eq!(deserialized.port, 8080);
    assert_eq!(deserialized.embedding_dim, Some(384));
}

