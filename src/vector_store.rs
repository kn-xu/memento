use crate::database::DatabaseClient;
use crate::embeddings::EmbeddingProvider;
use crate::types::Metadata;
use anyhow::Result;
use pgvector::Vector;
use sqlx::PgPool;
use std::sync::Arc;

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
                // Use pgvector
                let vector = Vector::from(embedding_vec.clone());
                sqlx::query("UPDATE memories SET embedding = $1::vector WHERE id = $2")
                    .bind(&vector)
                    .bind(memory_id)
                    .execute(pool)
                    .await?;
            }
            DatabaseClient::Sqlite(_) => {
                // Store as BLOB for SQLite
                let embedding_bytes: Vec<u8> = embedding_vec
                    .iter()
                    .flat_map(|f| f.to_le_bytes().to_vec())
                    .collect();
                
                // Note: sqlite-vss would require additional setup
                // For now, we'll store as BLOB and do cosine similarity in memory
                let conn = match &self.db {
                    DatabaseClient::Sqlite(conn) => Arc::clone(conn),
                    _ => unreachable!(),
                };
                
                tokio::task::spawn_blocking(move || {
                    conn.execute(
                        "UPDATE memories SET embedding = ?1 WHERE id = ?2",
                        rusqlite::params![embedding_bytes, memory_id],
                    )?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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
        filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let query_vector = Vector::from(query_embedding.to_vec());

        // Build query with proper parameter binding
        let mut query = sqlx::query_as::<_, (String, f64, Option<String>)>(
            "SELECT id, 1 - (embedding <=> $1::vector) as score, metadata
             FROM memories
             WHERE is_active = TRUE AND embedding IS NOT NULL"
        )
        .bind(&query_vector);

        if let Some(agent_id) = agent_id {
            // Note: This is simplified - in production you'd use a query builder
            // For now, we'll filter in memory after fetching
        }

        // Simplified: fetch all and filter in memory
        // In production, use sqlx query builder for proper parameterized queries
        let rows = sqlx::query(
            "SELECT id, 1 - (embedding <=> $1::vector) as score, metadata
             FROM memories
             WHERE is_active = TRUE AND embedding IS NOT NULL
             ORDER BY embedding <=> $1::vector
             LIMIT $2"
        )
        .bind(&query_vector)
        .bind(k as i64)
        .fetch_all(pool)
        .await?;

        let mut results = Vec::new();
        for row in rows {
            let memory_id: String = row.try_get("id")?;
            let score: f64 = row.try_get("score")?;
            let metadata_str: Option<String> = row.try_get("metadata")?;
            
            // Apply filters
            let mut should_include = true;
            if let Some(agent_id) = agent_id {
                // Would need to join or filter - simplified for now
            }
            if let Some(user_id) = user_id {
                // Would need to join or filter - simplified for now
            }
            
            if !should_include {
                continue;
            }
            
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
        filters: Metadata,
        agent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        let conn = match &self.db {
            DatabaseClient::Sqlite(conn) => Arc::clone(conn),
            _ => unreachable!(),
        };

        let mut where_clauses = vec!["is_active = 1".to_string()];
        let mut query_params: Vec<String> = vec![];

        if let Some(agent_id) = agent_id {
            where_clauses.push("agent_id = ?".to_string());
            query_params.push(agent_id.to_string());
        }

        if let Some(user_id) = user_id {
            where_clauses.push("user_id = ?".to_string());
            query_params.push(user_id.to_string());
        }

        // Note: Filter handling simplified - would need proper parameter binding
        let where_clause = where_clauses.join(" AND ");

        let query_str = format!(
            "SELECT id, embedding, metadata FROM memories WHERE {} AND embedding IS NOT NULL",
            where_clause
        );

        let rows = tokio::task::spawn_blocking(move || {
            // Build params vector for rusqlite
            let mut sqlite_params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            if let Some(agent_id) = agent_id {
                sqlite_params.push(agent_id);
            }
            if let Some(user_id) = user_id {
                sqlite_params.push(user_id);
            }
            
            let mut stmt = conn.prepare(&query_str)?;
            let rows = stmt.query_map(
                rusqlite::params_from_iter(sqlite_params.iter().map(|p| *p)),
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )?;
            
            let mut results = Vec::new();
            for row in rows {
                let (id, embedding_bytes, metadata_str) = row?;
                
                // Convert bytes back to f32 array
                let embedding: Vec<f32> = embedding_bytes
                    .chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let score = cosine_similarity(query_embedding, &embedding);
                
                let metadata: Metadata = if let Some(meta_str) = metadata_str {
                    serde_json::from_str(&meta_str).unwrap_or_default()
                } else {
                    Metadata::new()
                };

                results.push((id, score, metadata));
            }
            
            Ok::<Vec<(String, f64, Metadata)>, rusqlite::Error>(results)
        })
        .await??;

        // Sort by score and take top k
        let mut results: Vec<VectorSearchResult> = rows
            .into_iter()
            .map(|(memory_id, score, metadata)| VectorSearchResult {
                memory_id,
                score,
                metadata,
            })
            .collect();
        
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
            DatabaseClient::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                let memory_id = memory_id.to_string();
                tokio::task::spawn_blocking(move || {
                    conn.execute(
                        "UPDATE memories SET embedding = NULL WHERE id = ?1",
                        [&memory_id],
                    )?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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

