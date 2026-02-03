//! Database abstraction layer supporting SQLite and PostgreSQL.
//!
//! Provides a unified interface for memory storage operations across
//! both database backends.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool, Row, SqlitePool};

// =============================================================================
// Data Models
// =============================================================================

/// A raw interaction event before processing into searchable memory.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MemoryEvent {
    pub id: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: String,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: DateTime<Utc>,
    #[sqlx(default)]
    pub summarized_at: Option<DateTime<Utc>>,
}

/// A searchable memory record with embedding support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub memory_type: String,
    pub text: String,
    pub importance: f64,
    pub is_active: bool,
    pub supersedes_id: Option<String>,
    pub source_event_ids: Option<String>,
    pub metadata: Option<String>,
    pub last_accessed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

// =============================================================================
// Database Client
// =============================================================================

/// Database client supporting SQLite and PostgreSQL backends.
#[derive(Clone)]
pub enum DatabaseClient {
    Sqlite(SqlitePool),
    Postgres(PgPool),
}

impl DatabaseClient {
    /// Create a new database client with the specified embedding dimension.
    ///
    /// # Arguments
    /// * `database_url` - Connection URL (sqlite:// or postgresql://)
    /// * `embedding_dim` - Embedding vector dimension (must match your embedding provider)
    ///
    /// # Example
    /// ```ignore
    /// let db = DatabaseClient::new("sqlite::memory:", 384).await?;
    /// ```
    pub async fn new(database_url: &str, embedding_dim: usize) -> Result<Self> {
        if database_url.starts_with("postgresql://") || database_url.starts_with("postgres://") {
            let pool = PgPool::connect(database_url).await?;
            Self::init_postgres(&pool, Some(embedding_dim)).await?;
            Ok(Self::Postgres(pool))
        } else {
            let normalized_url = Self::normalize_sqlite_url(database_url);
            let pool = SqlitePool::connect(&normalized_url).await?;
            Self::init_sqlite(&pool).await?;
            Ok(Self::Sqlite(pool))
        }
    }

    /// Create a new database client using the embedding provider's dimension.
    /// This is the recommended constructor as it ensures the database schema
    /// matches the provider's embedding dimension automatically.
    ///
    /// # Example
    /// ```ignore
    /// let provider = get_embedding_provider(&config)?;
    /// let db = DatabaseClient::new_with_provider("sqlite::memory:", provider.as_ref()).await?;
    /// ```
    pub async fn new_with_provider(
        database_url: &str,
        provider: &dyn crate::embeddings::EmbeddingProvider,
    ) -> Result<Self> {
        Self::new(database_url, provider.dim()).await
    }

    #[doc(hidden)]
    pub fn normalize_sqlite_url(url: &str) -> String {
        let mut normalized = url.to_string();
        
        if normalized == ":memory:" || normalized == "sqlite::memory:" {
            return "sqlite::memory:".to_string();
        }
        
        if let Some((path, _query)) = normalized.split_once('?') {
            normalized = path.to_string();
        }
        
        if normalized.starts_with("sqlite://") {
            return normalized;
        } else if normalized.starts_with("sqlite::") {
            normalized = normalized.replacen("sqlite::", "sqlite://", 1);
            return normalized;
        }
        
        if !normalized.starts_with("sqlite://") {
            normalized = format!("sqlite://{}", normalized);
        }
        
        normalized
    }

    async fn init_sqlite(pool: &SqlitePool) -> Result<()> {
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summarized_at TIMESTAMP
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_agent_id ON memory_events(agent_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_user_id ON memory_events(user_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_session_id ON memory_events(session_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events(event_type)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_unsummarized ON memory_events(agent_id, user_id, session_id, summarized_at)")
            .execute(pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                memory_type TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                importance REAL DEFAULT 0.5,
                is_active BOOLEAN DEFAULT 1,
                supersedes_id TEXT,
                source_event_ids TEXT,
                metadata TEXT,
                last_accessed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (supersedes_id) REFERENCES memories(id)
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_agent_user_active_created ON memories(agent_id, user_id, is_active, created_at DESC)")
            .execute(pool)
            .await?;
        
        // Create memory_sources join table for provenance
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_sources (
                memory_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, event_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES memory_events(id) ON DELETE CASCADE
            )",
        )
        .execute(pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_sources_memory_id ON memory_sources(memory_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_sources_event_id ON memory_sources(event_id)")
            .execute(pool)
            .await?;

        Ok(())
    }

    async fn init_postgres(pool: &PgPool, embedding_dim: Option<usize>) -> Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summarized_at TIMESTAMP
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_agent_id ON memory_events(agent_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_user_id ON memory_events(user_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_session_id ON memory_events(session_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events(event_type)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_events_unsummarized ON memory_events(agent_id, user_id, session_id, summarized_at)")
            .execute(pool)
            .await?;

        // embedding_dim is required for Postgres (enforced in new())
        // Using expect() to fail fast if someone bypasses the guard
        let embedding_dim = embedding_dim.expect("embedding_dim required for Postgres");
        
        let create_table_sql = format!(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                memory_type TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({}),
                importance REAL DEFAULT 0.5,
                is_active BOOLEAN DEFAULT TRUE,
                supersedes_id TEXT,
                source_event_ids TEXT, 
                metadata TEXT,
                last_accessed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (supersedes_id) REFERENCES memories(id)
            )",
            embedding_dim
        );
        
        sqlx::query(&create_table_sql)
            .execute(pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_agent_user_active_created ON memories(agent_id, user_id, is_active, created_at DESC)")
            .execute(pool)
            .await?;
        
        // Create vector index for approximate similarity search
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
            .execute(pool)
            .await?;
        
        // Create memory_sources join table for provenance
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_sources (
                memory_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, event_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES memory_events(id) ON DELETE CASCADE
            )"
        )
        .execute(pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_sources_memory_id ON memory_sources(memory_id)")
            .execute(pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_sources_event_id ON memory_sources(event_id)")
            .execute(pool)
            .await?;

        Ok(())
    }

    pub async fn insert_event(&self, event: &MemoryEvent) -> Result<()> {
        match self {
            Self::Sqlite(pool) => {
                sqlx::query(
                    "INSERT INTO memory_events (id, agent_id, user_id, session_id, event_type, content, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )
                .bind(&event.id)
                .bind(&event.agent_id)
                .bind(&event.user_id)
                .bind(&event.session_id)
                .bind(&event.event_type)
                .bind(&event.content)
                .bind(&event.metadata)
                .execute(pool)
                .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query(
                    "INSERT INTO memory_events (id, agent_id, user_id, session_id, event_type, content, metadata)
                     VALUES ($1, $2, $3, $4, $5, $6, $7)",
                )
                .bind(&event.id)
                .bind(&event.agent_id)
                .bind(&event.user_id)
                .bind(&event.session_id)
                .bind(&event.event_type)
                .bind(&event.content)
                .bind(&event.metadata)
                .execute(pool)
                .await?;
            }
        }
        Ok(())
    }

    /// List unsummarized events for a given agent/user/session
    pub async fn list_unsummarized_events(
        &self,
        agent_id: &str,
        user_id: Option<&str>,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryEvent>> {
        match self {
            Self::Sqlite(pool) => {
                let mut sql = String::from(
                    "SELECT id, agent_id, user_id, session_id, event_type, content, metadata, created_at, summarized_at
                     FROM memory_events
                     WHERE agent_id = ?1 AND summarized_at IS NULL"
                );
                let mut bind_idx = 2;

                if user_id.is_some() {
                    sql.push_str(&format!(" AND user_id = ?{}", bind_idx));
                    bind_idx += 1;
                }

                if session_id.is_some() {
                    sql.push_str(&format!(" AND session_id = ?{}", bind_idx));
                    bind_idx += 1;
                }

                sql.push_str(&format!(" ORDER BY created_at ASC LIMIT ?{}", bind_idx));

                let mut query = sqlx::query_as::<_, MemoryEvent>(&sql).bind(agent_id);
                if let Some(uid) = user_id {
                    query = query.bind(uid);
                }
                if let Some(sid) = session_id {
                    query = query.bind(sid);
                }
                query = query.bind(limit as i64);

                let events = query.fetch_all(pool).await?;
                Ok(events)
            }
            Self::Postgres(pool) => {
                let mut sql = String::from(
                    "SELECT id, agent_id, user_id, session_id, event_type, content, metadata, created_at, summarized_at
                     FROM memory_events
                     WHERE agent_id = $1 AND summarized_at IS NULL"
                );
                let mut bind_idx = 2;

                if user_id.is_some() {
                    sql.push_str(&format!(" AND user_id = ${}", bind_idx));
                    bind_idx += 1;
                }

                if session_id.is_some() {
                    sql.push_str(&format!(" AND session_id = ${}", bind_idx));
                    bind_idx += 1;
                }

                sql.push_str(&format!(" ORDER BY created_at ASC LIMIT ${}", bind_idx));

                let mut query = sqlx::query_as::<_, MemoryEvent>(&sql).bind(agent_id);
                if let Some(uid) = user_id {
                    query = query.bind(uid);
                }
                if let Some(sid) = session_id {
                    query = query.bind(sid);
                }
                query = query.bind(limit as i64);

                let events = query.fetch_all(pool).await?;
                Ok(events)
            }
        }
    }

    /// Mark events as summarized by setting summarized_at timestamp
    /// Scoped by agent_id to prevent cross-agent contamination
    pub async fn mark_events_summarized(
        &self,
        agent_id: &str,
        event_ids: &[String],
    ) -> Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }

        match self {
            Self::Sqlite(pool) => {
                let placeholders: Vec<String> = (2..=event_ids.len() + 1).map(|i| format!("?{}", i)).collect();
                let sql = format!(
                    "UPDATE memory_events SET summarized_at = CURRENT_TIMESTAMP WHERE agent_id = ?1 AND id IN ({})",
                    placeholders.join(", ")
                );
                
                let mut query = sqlx::query(&sql).bind(agent_id);
                for id in event_ids {
                    query = query.bind(id);
                }
                query.execute(pool).await?;
            }
            Self::Postgres(pool) => {
                // sqlx accepts slices directly for array parameters
                sqlx::query("UPDATE memory_events SET summarized_at = NOW() WHERE agent_id = $1 AND id = ANY($2::text[])")
                    .bind(agent_id)
                    .bind(event_ids)
                    .execute(pool)
                    .await?;
            }
        }
        Ok(())
    }

    /// Insert a memory into the database.
    /// 
    /// Note: Embeddings are not stored via this function. They are added separately via
    /// `VectorStore::add()` which updates the embedding column directly.
    pub async fn insert_memory(&self, memory: &Memory) -> Result<()> {
        match self {
            Self::Sqlite(pool) => {
                sqlx::query(
                    "INSERT INTO memories (
                        id, agent_id, user_id, session_id, memory_type, text, embedding,
                        importance, is_active, supersedes_id, source_event_ids, metadata,
                        last_accessed_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .bind(&memory.id)
                .bind(&memory.agent_id)
                .bind(&memory.user_id)
                .bind(&memory.session_id)
                .bind(&memory.memory_type)
                .bind(&memory.text)
                .bind(&memory.importance)
                .bind(&memory.is_active)
                .bind(&memory.supersedes_id)
                .bind(&memory.source_event_ids)
                .bind(&memory.metadata)
                .bind(&memory.last_accessed_at)
                .execute(pool)
                .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query(
                    "INSERT INTO memories (
                        id, agent_id, user_id, session_id, memory_type, text, embedding,
                        importance, is_active, supersedes_id, source_event_ids, metadata,
                        last_accessed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, NULL, $7, $8, $9, $10, $11, $12)",
                )
                .bind(&memory.id)
                .bind(&memory.agent_id)
                .bind(&memory.user_id)
                .bind(&memory.session_id)
                .bind(&memory.memory_type)
                .bind(&memory.text)
                .bind(&memory.importance)
                .bind(&memory.is_active)
                .bind(&memory.supersedes_id)
                .bind(&memory.source_event_ids)
                .bind(&memory.metadata)
                .bind(&memory.last_accessed_at)
                .execute(pool)
                .await?;
            }
        }
        Ok(())
    }

    pub async fn get_memory(&self, id: &str) -> Result<Option<Memory>> {
        match self {
            Self::Sqlite(pool) => {
                let row = sqlx::query(
                    "SELECT id, agent_id, user_id, session_id, memory_type, text,
                            importance, is_active, supersedes_id, source_event_ids,
                            metadata, last_accessed_at, created_at
                     FROM memories WHERE id = ?1"
                )
                    .bind(id)
                    .fetch_optional(pool)
                    .await?;
                
                if let Some(row) = row {
                    Ok(Some(Self::sqlite_row_to_memory(&row)?))
                } else {
                    Ok(None)
                }
            }
            Self::Postgres(pool) => {
                let row = sqlx::query(
                    "SELECT id, agent_id, user_id, session_id, memory_type, text,
                            importance, is_active, supersedes_id, source_event_ids,
                            metadata, last_accessed_at, created_at
                     FROM memories WHERE id = $1"
                )
                    .bind(id)
                    .fetch_optional(pool)
                    .await?;
                
                if let Some(row) = row {
                    Ok(Some(Self::postgres_row_to_memory(&row)?))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub async fn get_memories_by_ids(&self, ids: &[String]) -> Result<std::collections::HashMap<String, Memory>> {
        use std::collections::HashMap;
        
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let mut result = HashMap::new();
        
        match self {
            Self::Sqlite(pool) => {
                // SQLite doesn't support IN with dynamic params easily, so we build the query
                let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
                let sql = format!(
                    "SELECT id, agent_id, user_id, session_id, memory_type, text,
                            importance, is_active, supersedes_id, source_event_ids,
                            metadata, last_accessed_at, created_at
                     FROM memories WHERE id IN ({})",
                    placeholders.join(", ")
                );
                
                let mut query = sqlx::query(&sql);
                for id in ids {
                    query = query.bind(id);
                }
                
                let rows = query.fetch_all(pool).await?;
                for row in rows {
                    let memory = Self::sqlite_row_to_memory(&row)?;
                    result.insert(memory.id.clone(), memory);
                }
            }
            Self::Postgres(pool) => {
                // PostgreSQL: use proper array binding (sqlx accepts slices directly)
                let rows = sqlx::query(
                    "SELECT id, agent_id, user_id, session_id, memory_type, text,
                            importance, is_active, supersedes_id, source_event_ids,
                            metadata, last_accessed_at, created_at
                     FROM memories WHERE id = ANY($1::text[])"
                )
                    .bind(ids)
                    .fetch_all(pool)
                    .await?;
                
                for row in rows {
                    let memory = Self::postgres_row_to_memory(&row)?;
                    result.insert(memory.id.clone(), memory);
                }
            }
        }
        
        Ok(result)
    }

    pub async fn update_memory_access(&self, id: &str) -> Result<()> {
        match self {
            Self::Sqlite(pool) => {
                sqlx::query("UPDATE memories SET last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?1")
                    .bind(id)
                    .execute(pool)
                    .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query("UPDATE memories SET last_accessed_at = CURRENT_TIMESTAMP WHERE id = $1")
                    .bind(id)
                    .execute(pool)
                    .await?;
            }
        }
        Ok(())
    }

    /// Boost importance when a memory is accessed, using diminishing returns.
    /// Also updates last_accessed_at timestamp.
    ///
    /// # Arguments
    /// * `id` - Memory ID to boost
    /// * `boost_amount` - Base boost value (will be scaled by diminishing returns)
    /// * `max_importance` - Maximum importance cap (must be > 0)
    ///
    /// # Edge Cases
    /// - If `max_importance <= 0`, returns early without updating (prevents division by zero)
    /// - If `boost_amount <= 0`, returns early (no-op)
    pub async fn boost_importance(&self, id: &str, boost_amount: f64, max_importance: f64) -> Result<()> {
        // Guard against division by zero in SQL and no-op cases
        if max_importance <= 0.0 || boost_amount <= 0.0 {
            return Ok(());
        }

        match self {
            Self::Sqlite(pool) => {
                // Diminishing returns formula: boost decreases as importance approaches max
                sqlx::query(
                    "UPDATE memories SET 
                        importance = MIN(?2, importance + ?3 * ((?2 - importance) / ?2)),
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?1"
                )
                    .bind(id)
                    .bind(max_importance)
                    .bind(boost_amount)
                    .execute(pool)
                    .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query(
                    "UPDATE memories SET 
                        importance = LEAST($2, importance + $3 * (($2 - importance) / $2)),
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = $1"
                )
                    .bind(id)
                    .bind(max_importance)
                    .bind(boost_amount)
                    .execute(pool)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn soft_delete_memory(&self, id: &str) -> Result<()> {
        match self {
            Self::Sqlite(pool) => {
                sqlx::query("UPDATE memories SET is_active = 0 WHERE id = ?1")
                    .bind(id)
                    .execute(pool)
                    .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query("UPDATE memories SET is_active = FALSE WHERE id = $1")
                    .bind(id)
                    .execute(pool)
                    .await?;
            }
        }
        Ok(())
    }

    /// Prune memories based on various criteria.
    /// Returns the number of memories that were (or would be, if dry_run) deleted.
    /// 
    /// This performs a HARD delete (permanent removal from database).
    pub async fn prune_memories(
        &self,
        older_than: Option<DateTime<Utc>>,
        importance_below: Option<f64>,
        never_accessed: bool,
        agent_id: Option<&str>,
        dry_run: bool,
    ) -> Result<u64> {
        // Build WHERE clauses
        let mut conditions = Vec::new();
        
        if older_than.is_some() {
            conditions.push("created_at < ?");
        }
        if importance_below.is_some() {
            conditions.push("importance < ?");
        }
        if never_accessed {
            conditions.push("last_accessed_at IS NULL");
        }
        if agent_id.is_some() {
            conditions.push("agent_id = ?");
        }
        
        if conditions.is_empty() {
            return Err(anyhow::anyhow!("At least one prune criterion is required"));
        }
        
        match self {
            Self::Sqlite(pool) => {
                // SQLite uses ?N placeholders
                let mut param_idx = 1;
                let mut where_parts = Vec::new();
                
                if older_than.is_some() {
                    where_parts.push(format!("created_at < ?{}", param_idx));
                    param_idx += 1;
                }
                if importance_below.is_some() {
                    where_parts.push(format!("importance < ?{}", param_idx));
                    param_idx += 1;
                }
                if never_accessed {
                    where_parts.push("last_accessed_at IS NULL".to_string());
                }
                if agent_id.is_some() {
                    where_parts.push(format!("agent_id = ?{}", param_idx));
                }
                
                let where_clause = where_parts.join(" AND ");
                
                if dry_run {
                    let count_sql = format!("SELECT COUNT(*) as count FROM memories WHERE {}", where_clause);
                    let mut query = sqlx::query_scalar::<_, i64>(&count_sql);
                    
                    if let Some(date) = older_than {
                        query = query.bind(date);
                    }
                    if let Some(imp) = importance_below {
                        query = query.bind(imp);
                    }
                    if let Some(agent) = agent_id {
                        query = query.bind(agent);
                    }
                    
                    let count = query.fetch_one(pool).await?;
                    Ok(count as u64)
                } else {
                    let delete_sql = format!("DELETE FROM memories WHERE {}", where_clause);
                    let mut query = sqlx::query(&delete_sql);
                    
                    if let Some(date) = older_than {
                        query = query.bind(date);
                    }
                    if let Some(imp) = importance_below {
                        query = query.bind(imp);
                    }
                    if let Some(agent) = agent_id {
                        query = query.bind(agent);
                    }
                    
                    let result = query.execute(pool).await?;
                    Ok(result.rows_affected())
                }
            }
            Self::Postgres(pool) => {
                // PostgreSQL uses $N placeholders
                let mut param_idx = 1;
                let mut where_parts = Vec::new();
                
                if older_than.is_some() {
                    where_parts.push(format!("created_at < ${}", param_idx));
                    param_idx += 1;
                }
                if importance_below.is_some() {
                    where_parts.push(format!("importance < ${}", param_idx));
                    param_idx += 1;
                }
                if never_accessed {
                    where_parts.push("last_accessed_at IS NULL".to_string());
                }
                if agent_id.is_some() {
                    where_parts.push(format!("agent_id = ${}", param_idx));
                }
                
                let where_clause = where_parts.join(" AND ");
                
                if dry_run {
                    let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {}", where_clause);
                    let mut query = sqlx::query_scalar::<_, i64>(&count_sql);
                    
                    if let Some(date) = older_than {
                        query = query.bind(date);
                    }
                    if let Some(imp) = importance_below {
                        query = query.bind(imp);
                    }
                    if let Some(agent) = agent_id {
                        query = query.bind(agent);
                    }
                    
                    let count = query.fetch_one(pool).await?;
                    Ok(count as u64)
                } else {
                    let delete_sql = format!("DELETE FROM memories WHERE {}", where_clause);
                    let mut query = sqlx::query(&delete_sql);
                    
                    if let Some(date) = older_than {
                        query = query.bind(date);
                    }
                    if let Some(imp) = importance_below {
                        query = query.bind(imp);
                    }
                    if let Some(agent) = agent_id {
                        query = query.bind(agent);
                    }
                    
                    let result = query.execute(pool).await?;
                    Ok(result.rows_affected())
                }
            }
        }
    }

    /// This populates the memory_sources join table for tracking event provenance.
    /// 
    /// Note: For large batches (hundreds+ event_ids), consider optimizing with batch inserts:
    /// - SQLite: VALUES (?1, ?2), (?1, ?3), ...
    /// - Postgres: SELECT $1, unnest($2::text[])
    /// 
    /// Current implementation uses per-event inserts, which is fine for typical use cases.
    pub async fn insert_memory_sources(
        &self,
        memory_id: &str,
        event_ids: &[String],
    ) -> Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }

        match self {
            Self::Sqlite(pool) => {
                for event_id in event_ids {
                    sqlx::query(
                        "INSERT OR IGNORE INTO memory_sources (memory_id, event_id) VALUES (?1, ?2)"
                    )
                    .bind(memory_id)
                    .bind(event_id)
                    .execute(pool)
                    .await?;
                }
            }
            Self::Postgres(pool) => {
                for event_id in event_ids {
                    sqlx::query(
                        "INSERT INTO memory_sources (memory_id, event_id) VALUES ($1, $2) ON CONFLICT DO NOTHING"
                    )
                    .bind(memory_id)
                    .bind(event_id)
                    .execute(pool)
                    .await?;
                }
            }
        }
        Ok(())
    }

    /// Get the source event IDs for a memory (provenance tracking).
    pub async fn get_memory_sources(&self, memory_id: &str) -> Result<Vec<String>> {
        match self {
            Self::Sqlite(pool) => {
                let rows: Vec<(String,)> = sqlx::query_as(
                    "SELECT event_id FROM memory_sources WHERE memory_id = ?1"
                )
                .bind(memory_id)
                .fetch_all(pool)
                .await?;
                Ok(rows.into_iter().map(|(id,)| id).collect())
            }
            Self::Postgres(pool) => {
                let rows: Vec<(String,)> = sqlx::query_as(
                    "SELECT event_id FROM memory_sources WHERE memory_id = $1"
                )
                .bind(memory_id)
                .fetch_all(pool)
                .await?;
                Ok(rows.into_iter().map(|(id,)| id).collect())
            }
        }
    }

    // -------------------------------------------------------------------------
    // Row Conversion Helpers
    // -------------------------------------------------------------------------
    // These functions are nearly identical but must be separate due to sqlx's
    // type system requiring concrete row types. The only difference is how
    // `is_active` is handled (SQLite returns i32, Postgres returns bool).

    fn sqlite_row_to_memory(row: &sqlx::sqlite::SqliteRow) -> Result<Memory> {
        let is_active: i32 = row.try_get("is_active").unwrap_or(1);
        Self::build_memory_from_row_data(
            row.try_get("id")?,
            row.try_get("agent_id")?,
            row.try_get("user_id").ok(),
            row.try_get("session_id").ok(),
            row.try_get("memory_type")?,
            row.try_get("text")?,
            row.try_get("importance").unwrap_or(0.5),
            is_active != 0,
            row.try_get("supersedes_id").ok(),
            row.try_get("source_event_ids").ok(),
            row.try_get("metadata").ok(),
            row.try_get("last_accessed_at").ok(),
            row.try_get("created_at")?,
        )
    }

    fn postgres_row_to_memory(row: &sqlx::postgres::PgRow) -> Result<Memory> {
        Self::build_memory_from_row_data(
            row.try_get("id")?,
            row.try_get("agent_id")?,
            row.try_get("user_id").ok(),
            row.try_get("session_id").ok(),
            row.try_get("memory_type")?,
            row.try_get("text")?,
            row.try_get("importance").unwrap_or(0.5),
            row.try_get("is_active").unwrap_or(true),
            row.try_get("supersedes_id").ok(),
            row.try_get("source_event_ids").ok(),
            row.try_get("metadata").ok(),
            row.try_get("last_accessed_at").ok(),
            row.try_get("created_at")?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_memory_from_row_data(
        id: String,
        agent_id: String,
        user_id: Option<String>,
        session_id: Option<String>,
        memory_type: String,
        text: String,
        importance: f64,
        is_active: bool,
        supersedes_id: Option<String>,
        source_event_ids: Option<String>,
        metadata: Option<String>,
        last_accessed_at: Option<DateTime<Utc>>,
        created_at: DateTime<Utc>,
    ) -> Result<Memory> {
        Ok(Memory {
            id,
            agent_id,
            user_id,
            session_id,
            memory_type,
            text,
            importance,
            is_active,
            supersedes_id,
            source_event_ids,
            metadata,
            last_accessed_at,
            created_at,
        })
    }
}
