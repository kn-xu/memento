//! Tests for embeddings module

use memento::embeddings::{
    get_embedding_provider, DummyEmbeddingProvider, EmbeddingProvider, OpenAIEmbeddingProvider,
};

#[tokio::test]
async fn test_dummy_provider_embed() {
    let provider = DummyEmbeddingProvider::new(None, Some(384));
    let embedding = provider.embed("test text").await.unwrap();

    assert_eq!(embedding.len(), 384);
    // Check normalization (L2 norm should be ~1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_dummy_provider_deterministic() {
    let provider = DummyEmbeddingProvider::new(None, Some(384));
    let embedding1 = provider.embed("same text").await.unwrap();
    let embedding2 = provider.embed("same text").await.unwrap();

    assert_eq!(embedding1, embedding2);
}

#[tokio::test]
async fn test_dummy_provider_different_texts_different_embeddings() {
    let provider = DummyEmbeddingProvider::new(None, Some(384));
    let embedding1 = provider.embed("text one").await.unwrap();
    let embedding2 = provider.embed("text two").await.unwrap();

    assert_ne!(embedding1, embedding2);
}

#[tokio::test]
async fn test_dummy_provider_batch_embed() {
    let provider = DummyEmbeddingProvider::new(None, Some(384));
    let texts = vec![
        "text1".to_string(),
        "text2".to_string(),
        "text3".to_string(),
    ];
    let embeddings = provider.embed_batch(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 384);
    }
}

#[tokio::test]
async fn test_dummy_provider_custom_dimension() {
    let provider = DummyEmbeddingProvider::new(None, Some(512));
    assert_eq!(provider.dim(), 512);

    let embedding = provider.embed("test").await.unwrap();
    assert_eq!(embedding.len(), 512);
}

#[tokio::test]
async fn test_dummy_provider_default_dimension() {
    let provider = DummyEmbeddingProvider::new(None, None);
    assert_eq!(provider.dim(), 384);
}

#[test]
fn test_get_embedding_provider_local() {
    let provider = get_embedding_provider("local", None, None, Some(384)).unwrap();
    assert_eq!(provider.dim(), 384);
}

#[test]
fn test_get_embedding_provider_openai_requires_key() {
    let result = get_embedding_provider("openai", None, None, None);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("API key required"));
}

#[test]
fn test_openai_provider_dimension_text_embedding_3_small() {
    let provider = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("text-embedding-3-small".to_string()),
        None,
    )
    .unwrap();
    assert_eq!(provider.dim(), 1536);
}

#[test]
fn test_openai_provider_dimension_text_embedding_3_large() {
    let provider = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("text-embedding-3-large".to_string()),
        None,
    )
    .unwrap();
    assert_eq!(provider.dim(), 3072);
}

#[test]
fn test_openai_provider_custom_dimension() {
    let provider = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("text-embedding-3-small".to_string()),
        Some(512),
    )
    .unwrap();
    assert_eq!(provider.dim(), 512);
}

#[test]
fn test_openai_provider_rejects_dimension_exceeding_base() {
    let result = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("text-embedding-3-small".to_string()),
        Some(2000), // Exceeds 1536
    );
    assert!(result.is_err());
}

#[test]
fn test_openai_provider_unknown_model_without_dimension_fails() {
    let result = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("unknown-model".to_string()),
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_openai_provider_unknown_model_with_dimension_succeeds() {
    let provider = OpenAIEmbeddingProvider::new(
        "test-key".to_string(),
        Some("unknown-model".to_string()),
        Some(768),
    )
    .unwrap();
    assert_eq!(provider.dim(), 768);
}

