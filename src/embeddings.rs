use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct LocalEmbeddingProvider {
    model_name: String,
}

impl LocalEmbeddingProvider {
    pub fn new(model_name: Option<String>) -> Self {
        Self {
            model_name: model_name.unwrap_or_else(|| "Xenova/all-MiniLM-L6-v2".to_string()),
        }
    }

    // Note: For a production implementation, you'd use candle-transformers
    // This is a simplified version that would need actual model loading
    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Placeholder: In production, load and run the transformer model
        // For now, return a dummy embedding
        // TODO: Implement actual transformer inference using candle
        
        // Dummy embedding of size 384 (typical for MiniLM)
        let mut embedding = vec![0.0f32; 384];
        let hash = text.len() as u64;
        for i in 0..384 {
            embedding[i] = ((hash + i as u64) % 1000) as f32 / 1000.0;
        }
        
        Ok(embedding)
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.get_embedding(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.get_embedding(text).await?);
        }
        Ok(results)
    }
}

pub struct OpenAIEmbeddingProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String, model: Option<String>) -> Self {
        Self {
            api_key,
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "input": text
            }))
            .send()
            .await?;

        let response_data: serde_json::Value = response.json().await?;
        let embedding = response_data["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "input": texts
            }))
            .send()
            .await?;

        let response_data: serde_json::Value = response.json().await?;
        let embeddings: Result<Vec<Vec<f32>>> = response_data["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
                    .into()
            })
            .collect();

        embeddings
    }
}

pub fn get_embedding_provider(
    provider_type: &str,
    api_key: Option<String>,
    model: Option<String>,
) -> Result<Box<dyn EmbeddingProvider + Send + Sync>> {
    match provider_type {
        "openai" => {
            let api_key = api_key.ok_or_else(|| anyhow::anyhow!("OpenAI API key required"))?;
            Ok(Box::new(OpenAIEmbeddingProvider::new(api_key, model)))
        }
        _ => Ok(Box::new(LocalEmbeddingProvider::new(model))),
    }
}

