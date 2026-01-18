//! Embedding providers for generating vector embeddings from text.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use sha2::{Sha256, Digest};

// =============================================================================
// Shared Utilities
// =============================================================================

/// Normalize an embedding vector to unit length (L2 normalization).
/// This ensures consistent cosine similarity calculations.
#[inline]
pub fn normalize_embedding(embedding: &mut [f32]) {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

// =============================================================================
// Traits
// =============================================================================

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// =============================================================================
// Dummy Provider (for testing)
// =============================================================================

pub struct DummyEmbeddingProvider {
    dim: usize,
}

impl DummyEmbeddingProvider {
    pub fn new(_model_name: Option<String>, dimensions: Option<usize>) -> Self {
        eprintln!("⚠️  Warning: Using DummyEmbeddingProvider - embeddings are deterministic hashes, not semantic. This is for testing only.");
        Self {
            dim: dimensions.unwrap_or(384),
        }
    }

    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dim];
        let hash = Sha256::digest(text.as_bytes());
        for i in 0..self.dim {
            embedding[i] = hash[i % hash.len()] as f32 / 255.0;
        }
        normalize_embedding(&mut embedding);
        embedding
    }
}

#[async_trait]
impl EmbeddingProvider for DummyEmbeddingProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.generate_embedding(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }
}

// =============================================================================
// OpenAI Provider
// =============================================================================

pub struct OpenAIEmbeddingProvider {
    api_key: String,
    model: String,
    dimensions: Option<usize>,
    client: reqwest::Client,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String, model: Option<String>, dimensions: Option<usize>) -> Result<Self> {
        let model = model.unwrap_or_else(|| "text-embedding-3-small".to_string());
        
        if dimensions.is_none() && model != "text-embedding-3-small" && model != "text-embedding-3-large" {
            return Err(anyhow::anyhow!(
                "Unknown embedding model '{}'; set dimensions explicitly via MEMENTO_EMBEDDING_DIM",
                model
            ));
        }

        if let Some(d) = dimensions {
            let base = match model.as_str() {
                "text-embedding-3-small" => 1536,
                "text-embedding-3-large" => 3072,
                _ => d, // unknown already rejected unless d provided
            };
            if d > base {
                return Err(anyhow::anyhow!(
                    "Requested dimensions {} exceed model base dim {}",
                    d, base
                ));
            }
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            api_key,
            model,
            dimensions,
            client,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn dim(&self) -> usize {
        if let Some(d) = self.dimensions {
            return d;
        }
        match self.model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => unreachable!("Unknown model should be caught in new()"),
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request_body = self.build_request_body(json!(text));
        let response = self.post_with_retry(request_body).await?;
        let data = self.parse_response_data(response).await?;
        let mut embedding = self.parse_embedding_from_json(&data[0]["embedding"])?;
        normalize_embedding(&mut embedding);
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request_body = self.build_request_body(json!(texts));
        let response = self.post_with_retry(request_body).await?;
        let data = self.parse_response_data(response).await?;

        let mut items: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let index = item["index"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Missing index in OpenAI response"))? as usize;
                
                if index >= texts.len() {
                    return Err(anyhow::anyhow!(
                        "Index {} out of bounds for batch size {}",
                        index,
                        texts.len()
                    ));
                }
                
                let mut embedding = self.parse_embedding_from_json(&item["embedding"])?;
                normalize_embedding(&mut embedding);
                Ok((index, embedding))
            })
            .collect::<Result<Vec<_>>>()?;

        if items.len() != texts.len() {
            return Err(anyhow::anyhow!(
                "Batch embedding count mismatch: got {}, expected {}",
                items.len(),
                texts.len()
            ));
        }

        items.sort_by_key(|(idx, _)| *idx);

        let indices: std::collections::HashSet<usize> = items.iter().map(|(i, _)| *i).collect();
        if indices.len() != texts.len() {
            return Err(anyhow::anyhow!("Duplicate or missing indices in batch response"));
        }

        Ok(items.into_iter().map(|(_, emb)| emb).collect())
    }
}

impl OpenAIEmbeddingProvider {
    async fn post_with_retry(&self, body: serde_json::Value) -> Result<reqwest::Response> {
        let mut last_err = None;
        for attempt in 0..3 {
            let resp = self.client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) => {
                    let status = r.status();
                    let txt = r.text().await.unwrap_or_default();
                    last_err = Some(anyhow::anyhow!("OpenAI {}: {}", status, txt));
                    if status.as_u16() != 429 && !status.is_server_error() {
                        break;
                    }
                }
                Err(e) => last_err = Some(e.into()),
            }

            if attempt < 2 {
                tokio::time::sleep(std::time::Duration::from_millis(200 * (1 << attempt))).await;
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Unknown OpenAI error")))
    }

    fn build_request_body(&self, input: serde_json::Value) -> serde_json::Value {
        let mut body = json!({
            "model": &self.model,
            "input": input
        });
        if let Some(dim) = self.dimensions {
            body["dimensions"] = json!(dim);
        }
        body
    }

    async fn parse_response_data(&self, response: reqwest::Response) -> Result<Vec<serde_json::Value>> {
        let response_data: serde_json::Value = response.json().await?;
        let data = response_data["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing data array in embeddings response"))?;
        
        if data.is_empty() {
            return Err(anyhow::anyhow!("Embeddings response data array was empty"));
        }

        Ok(data.clone())
    }

    fn parse_embedding_from_json(&self, embedding_json: &serde_json::Value) -> Result<Vec<f32>> {
        let embedding: Vec<f32> = embedding_json
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
            .iter()
            .map(|v| v.as_f64().ok_or_else(|| anyhow::anyhow!("Non-numeric embedding value")))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|f| f as f32)
            .collect();

        let expected = self.dim();
        if embedding.len() != expected {
            return Err(anyhow::anyhow!(
                "Embedding dim mismatch: got {}, expected {}",
                embedding.len(),
                expected
            ));
        }

        Ok(embedding)
    }
}

// =============================================================================
// Factory
// =============================================================================

/// Create an embedding provider based on the provider type.
pub fn get_embedding_provider(
    provider_type: &str,
    api_key: Option<String>,
    model: Option<String>,
    dimensions: Option<usize>,
) -> Result<Box<dyn EmbeddingProvider + Send + Sync>> {
    match provider_type {
        "openai" => {
            let api_key = api_key.ok_or_else(|| anyhow::anyhow!("OpenAI API key required"))?;
            Ok(Box::new(OpenAIEmbeddingProvider::new(api_key, model, dimensions)?))
        }
        _ => Ok(Box::new(DummyEmbeddingProvider::new(model, dimensions))),
    }
}
