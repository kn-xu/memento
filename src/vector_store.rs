use crate::database::DatabaseClient;
use crate::embeddings::EmbeddingProvider;
use crate::types::Metadata;
use anyhow::Result;
use pgvector::Vector;
use sqlx::{PgPool, QueryBuilder, Row};

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

    fn apply_filters<'q, DB: sqlx::Database>(
        query_builder: &mut QueryBuilder<'q, DB>,
        agent_id: Option<&'q str>,
        user_id: Option<&'q str>,
        filters: &'q Metadata,
    ) where
        &'q str: sqlx::Encode<'q, DB> + sqlx::Type<DB>,
        f64: sqlx::Encode<'q, DB> + sqlx::Type<DB>,
    {
        for (opt_val, field) in [
            (agent_id, "agent_id"),
            (user_id, "user_id"),
            (filters.get("memory_type").and_then(|v| v.as_str()), "memory_type"),
        ] {
            if let Some(val) = opt_val {
                query_builder.push(" AND ");
                query_builder.push(field);
                query_builder.push(" = ");
                query_builder.push_bind(val);
            }
        }

        if let Some(min_importance) = filters.get("min_importance").and_then(|v| v.as_f64()) {
            query_builder.push(" AND importance >= ");
            query_builder.push_bind(min_importance);
        }
    }

    fn parse_metadata(metadata_str: Option<String>) -> Metadata {
        metadata_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(Metadata::new)
    }

    fn build_result_from_data(memory_id: String, metadata_str: Option<String>, score: f64) -> VectorSearchResult {
        let metadata = Self::parse_metadata(metadata_str);
        VectorSearchResult {
            memory_id,
            score,
            metadata,
        }
    }


    pub async fn add(
        &self,
        memory_id: &str,
        text: &str,
        embedding: Option<Vec<f32>>,
        metadata: Metadata,
    ) -> Result<()> {
        let embedding_vec = if let Some(emb) = embedding {
            emb
        } else {
            self.embedding_provider.embed(text).await?
        };

        let expected_dim = self.embedding_provider.dim();
        if embedding_vec.len() != expected_dim {
            return Err(anyhow::anyhow!(
                "Embedding dimension {} does not match expected {}",
                embedding_vec.len(),
                expected_dim
            ));
        }

        let metadata_json = if metadata.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&metadata)?)
        };

        match &self.db {
            DatabaseClient::Postgres(pool) => {
                let vector = Vector::from(embedding_vec);
                if let Some(meta_json) = &metadata_json {
                    sqlx::query("UPDATE memories SET embedding = $1::vector, metadata = $2 WHERE id = $3")
                        .bind(vector)
                        .bind(meta_json)
                        .bind(memory_id)
                        .execute(pool)
                        .await?;
                } else {
                    sqlx::query("UPDATE memories SET embedding = $1::vector WHERE id = $2")
                        .bind(vector)
                        .bind(memory_id)
                        .execute(pool)
                        .await?;
                }
            }
            DatabaseClient::Sqlite(pool) => {
                let mut embedding_bytes = Vec::with_capacity(embedding_vec.len() * 4);
                for f in &embedding_vec {
                    embedding_bytes.extend_from_slice(&f.to_le_bytes());
                }
                
                if let Some(meta_json) = &metadata_json {
                    sqlx::query("UPDATE memories SET embedding = ?1, metadata = ?2 WHERE id = ?3")
                        .bind(&embedding_bytes)
                        .bind(meta_json)
                        .bind(memory_id)
                        .execute(pool)
                        .await?;
                } else {
                    sqlx::query("UPDATE memories SET embedding = ?1 WHERE id = ?2")
                        .bind(&embedding_bytes)
                        .bind(memory_id)
                        .execute(pool)
                        .await?;
                }
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

    /// Search by pre-computed embedding vector.
    /// Used for novelty scoring to avoid recomputing embeddings.
    pub async fn search_by_embedding(
        &self,
        embedding: &[f32],
        k: usize,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let filters = Metadata::new();
        match &self.db {
            DatabaseClient::Postgres(pool) => {
                self.search_pgvector(pool, embedding, k, filters, Some(agent_id), user_id)
                    .await
            }
            DatabaseClient::Sqlite(_) => {
                self.search_sqlite(embedding, k, filters, Some(agent_id), user_id).await
            }
        }
    }

    /// Compute embedding for text without storing it.
    /// Used to get embedding before deciding on importance.
    pub async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding_provider.embed(text).await
    }

    async fn search_pgvector(
        &self,
        pool: &PgPool,
        query_embedding: &[f32],
        k: usize,
        filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let query_vector = Vector::from(query_embedding.to_vec());

        let mut query_builder: QueryBuilder<'_, sqlx::Postgres> = QueryBuilder::new(
            "SELECT id, 1 - (embedding <=> "
        );
        query_builder.push_bind(query_vector.clone());
        query_builder.push("::vector) AS score, metadata FROM memories WHERE is_active = TRUE AND embedding IS NOT NULL");

        Self::apply_filters(&mut query_builder, agent_id, user_id, &filters);

        query_builder.push(" ORDER BY embedding <=> ");
        query_builder.push_bind(query_vector);
        query_builder.push(" LIMIT ");
        query_builder.push_bind(k as i64);

        let query = query_builder.build();
        let rows = query.fetch_all(pool).await?;

        let results: Result<Vec<_>> = rows
            .iter()
            .map(|row| {
                let score: f64 = row.try_get("score")?;
                let memory_id: String = row.try_get("id")?;
                let metadata_str: Option<String> = row.try_get("metadata").ok();
                Ok(Self::build_result_from_data(memory_id, metadata_str, score))
            })
            .collect();
        results
    }

    async fn search_sqlite(
        &self,
        query_embedding: &[f32],
        k: usize,
        filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let pool = match &self.db {
            DatabaseClient::Sqlite(pool) => pool,
            _ => unreachable!(),
        };

        let fetch_limit = (k * 10).max(100).min(5000);
        let mut query_builder: QueryBuilder<'_, sqlx::Sqlite> = QueryBuilder::new(
            "SELECT id, embedding, metadata FROM memories WHERE is_active = 1 AND embedding IS NOT NULL"
        );

        Self::apply_filters(&mut query_builder, agent_id, user_id, &filters);

        query_builder.push(" ORDER BY created_at DESC LIMIT ");
        query_builder.push_bind(fetch_limit as i64);

        let query = query_builder.build();
        let rows = query.fetch_all(pool).await?;

        let mut results = Vec::new();
        for row in rows {
            let embedding_bytes: Vec<u8> = match row.try_get("embedding") {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            
            if embedding_bytes.len() % 4 != 0 {
                eprintln!("⚠️  Skipping corrupted embedding: byte length {} is not divisible by 4 (memory_id: {:?})", 
                    embedding_bytes.len(),
                    row.try_get::<String, _>("id").ok()
                );
                continue;
            }

            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            if embedding.len() != query_embedding.len() {
                eprintln!("⚠️  Skipping embedding with dimension mismatch: got {}, expected {} (memory_id: {:?})",
                    embedding.len(),
                    query_embedding.len(),
                    row.try_get::<String, _>("id").ok()
                );
                continue;
            }

            let score = cosine_similarity(query_embedding, &embedding);
            let memory_id: String = match row.try_get("id") {
                Ok(id) => id,
                Err(_) => continue,
            };
            let metadata_str: Option<String> = row.try_get("metadata").ok();
            results.push(Self::build_result_from_data(memory_id, metadata_str, score));
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

/// Computes cosine similarity between two vectors.
/// 
/// # Returns
/// - Similarity score in range [-1.0, 1.0] for valid inputs
/// - 0.0 for edge cases: different lengths, zero vectors, or NaN values
/// 
/// # Edge Cases Handled
/// - Different vector lengths → 0.0
/// - Zero-magnitude vectors → 0.0
/// - NaN values in either vector → 0.0
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        
        // Check for NaN - if any value is NaN, return 0.0
        if !ai.is_finite() || !bi.is_finite() {
            return 0.0;
        }
        
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let result = dot_product / (norm_a.sqrt() * norm_b.sqrt());
    
    // Final check: if result is NaN (shouldn't happen, but defensive)
    if result.is_finite() {
        result
    } else {
        0.0
    }
}

