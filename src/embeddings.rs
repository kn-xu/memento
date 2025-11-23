use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct LocalEmbeddingProvider;

impl LocalEmbeddingProvider {
    pub fn new(_model_name: Option<String>) -> Self {
        Self
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>> {
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
    fn dim(&self) -> usize {
        384
    }

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
    dimensions: Option<usize>,
    client: reqwest::Client,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String, model: Option<String>, dimensions: Option<usize>) -> Self {
        Self {
            api_key,
            model: model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
            dimensions,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn dim(&self) -> usize {
        if let Some(d) = self.dimensions {
            return d;
        }
        if self.model.contains("text-embedding-3-large") {
            3072
        } else {
            1536
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut request_body = json!({
            "model": self.model,
            "input": text
        });
        
        if let Some(dim) = self.dimensions {
            request_body["dimensions"] = json!(dim);
        }
        
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
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
        let mut request_body = json!({
            "model": self.model,
            "input": texts
        });
        
        if let Some(dim) = self.dimensions {
            request_body["dimensions"] = json!(dim);
        }
        
        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let response_data: serde_json::Value = response.json().await?;
        let embeddings: Result<Vec<Vec<f32>>> = response_data["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?
            .iter()
            .map(|item| {
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                Ok(embedding)
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
            Ok(Box::new(OpenAIEmbeddingProvider::new(api_key, model, None)))
        }
        _ => Ok(Box::new(LocalEmbeddingProvider::new(model))),
    }
}

