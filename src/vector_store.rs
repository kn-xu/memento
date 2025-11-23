use crate::database::DatabaseClient;
use crate::embeddings::EmbeddingProvider;
use crate::types::Metadata;
use anyhow::Result;
use pgvector::Vector;
use sqlx::{PgPool, Row};

pub struct VectorStore {
    db: DatabaseClient,
    embedding_provider: Box<dyn EmbeddingProvider + Send + Sync>,
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub memory_id: String,
    pub score: f64,
    pub metadata: Metadata,
}

impl VectorStore {
    pub fn new(db: DatabaseClient, embedding_provider: Box<dyn EmbeddingProvider + Send + Sync>) -> Self {
        Self {
            db,
            embedding_provider,
        }
    }

    pub async fn add(
        &self,
        memory_id: &str,
        text: &str,
        embedding: Option<Vec<f32>>,
        _metadata: Metadata,
    ) -> Result<()> {
        let embedding_vec = if let Some(emb) = embedding {
            emb
        } else {
            self.embedding_provider.embed(text).await?
        };

        match &self.db {
            DatabaseClient::Postgres(pool) => {
                let vector = Vector::from(embedding_vec.clone());
                sqlx::query("UPDATE memories SET embedding = $1::vector WHERE id = $2")
                    .bind(vector)
                    .bind(memory_id)
                    .execute(pool)
                    .await?;
            }
            DatabaseClient::Sqlite(pool) => {
                let embedding_bytes: Vec<u8> = embedding_vec
                    .iter()
                    .flat_map(|f| f.to_le_bytes().to_vec())
                    .collect();
                sqlx::query("UPDATE memories SET embedding = ?1 WHERE id = ?2")
                    .bind(&embedding_bytes)
                    .bind(memory_id)
                    .execute(pool)
                    .await?;
            }
        }

        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        k: usize,
        filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let query_embedding = self.embedding_provider.embed(query).await?;

        match &self.db {
            DatabaseClient::Postgres(pool) => {
                self.search_pgvector(pool, &query_embedding, k, filters, agent_id, user_id)
                    .await
            }
            DatabaseClient::Sqlite(_) => {
                self.search_sqlite(&query_embedding, k, filters, agent_id, user_id).await
            }
        }
    }

    async fn search_pgvector(
        &self,
        pool: &PgPool,
        query_embedding: &[f32],
        k: usize,
        _filters: Metadata,
        _agent_id: Option<&str>,
        _user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let query_vector = Vector::from(query_embedding.to_vec());

        let rows = sqlx::query(
            "SELECT id, 1 - (embedding <=> $1::vector) as score, metadata
             FROM memories
             WHERE is_active = TRUE AND embedding IS NOT NULL
             ORDER BY embedding <=> $1::vector
             LIMIT $2"
        )
        .bind(query_vector)
        .bind(k as i64)
        .fetch_all(pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            let memory_id: String = row.try_get(&"id")?;
            let score: f64 = row.try_get(&"score")?;
            let metadata_str: Option<String> = row.try_get(&"metadata")?;
            
            let metadata: Metadata = if let Some(meta_str) = metadata_str {
                serde_json::from_str(&meta_str).unwrap_or_default()
            } else {
                Metadata::new()
            };

            results.push(VectorSearchResult {
                memory_id,
                score,
                metadata,
            });
        }

        Ok(results)
    }

    async fn search_sqlite(
        &self,
        query_embedding: &[f32],
        k: usize,
        _filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let pool = match &self.db {
            DatabaseClient::Sqlite(pool) => pool,
            _ => unreachable!(),
        };

        let query = match (agent_id, user_id) {
            (Some(agent_id), Some(user_id)) => {
                sqlx::query("SELECT id, embedding, metadata FROM memories 
                             WHERE is_active = 1 AND embedding IS NOT NULL AND agent_id = ?1 AND user_id = ?2")
                    .bind(agent_id)
                    .bind(user_id)
            },
            (Some(agent_id), None) => {
                sqlx::query("SELECT id, embedding, metadata FROM memories 
                             WHERE is_active = 1 AND embedding IS NOT NULL AND agent_id = ?1")
                    .bind(agent_id)
            },
            (None, Some(user_id)) => {
                sqlx::query("SELECT id, embedding, metadata FROM memories 
                             WHERE is_active = 1 AND embedding IS NOT NULL AND user_id = ?1")
                    .bind(user_id)
            },
            (None, None) => {
                sqlx::query("SELECT id, embedding, metadata FROM memories 
                             WHERE is_active = 1 AND embedding IS NOT NULL")
            },
        };

        let rows = query.fetch_all(pool).await?;

        let mut results = Vec::new();
        for row in rows {
            let memory_id: String = row.try_get(&"id")?;
            let embedding_bytes: Vec<u8> = row.try_get(&"embedding")?;
            let metadata_str: Option<String> = row.try_get(&"metadata")?;
            
            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let score = cosine_similarity(query_embedding, &embedding);
            
            let metadata: Metadata = if let Some(meta_str) = metadata_str {
                serde_json::from_str(&meta_str).unwrap_or_default()
            } else {
                Metadata::new()
            };

            results.push(VectorSearchResult {
                memory_id,
                score,
                metadata,
            });
        }
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    pub async fn delete(&self, memory_id: &str) -> Result<()> {
        match &self.db {
            DatabaseClient::Postgres(pool) => {
                sqlx::query("UPDATE memories SET embedding = NULL WHERE id = $1")
                    .bind(memory_id)
                    .execute(pool)
                    .await?;
            }
            DatabaseClient::Sqlite(pool) => {
                sqlx::query("UPDATE memories SET embedding = NULL WHERE id = ?1")
                    .bind(memory_id)
                    .execute(pool)
                    .await?;
            }
        }
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += (a[i] as f64) * (b[i] as f64);
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

