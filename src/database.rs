use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, SqlitePool, Row};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub id: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub event_type: String,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub memory_type: String,
    pub text: String,
    pub embedding: Option<Vec<u8>>,
    pub importance: f64,
    pub is_active: bool,
    pub supersedes_id: Option<String>,
    pub source_event_ids: Option<String>,
    pub metadata: Option<String>,
    pub last_accessed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Clone)]
pub enum DatabaseClient {
    Sqlite(SqlitePool),
    Postgres(PgPool),
}

impl DatabaseClient {
    pub async fn new(database_url: &str) -> Result<Self> {
        if database_url.starts_with("postgresql://") || database_url.starts_with("postgres://") {
            let pool = PgPool::connect(database_url).await?;
            Self::init_postgres(&pool).await?;
            Ok(Self::Postgres(pool))
        } else {
            let normalized_url = Self::normalize_sqlite_url(database_url);
            let pool = SqlitePool::connect(&normalized_url).await?;
            Self::init_sqlite(&pool).await?;
            Ok(Self::Sqlite(pool))
        }
    }

    fn normalize_sqlite_url(url: &str) -> String {
        let mut normalized = url.to_string();
        
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                expires_at TIMESTAMP,
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

        Ok(())
    }

    async fn init_postgres(pool: &PgPool) -> Result<()> {
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        let embedding_dim = 384;
        
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
                expires_at TIMESTAMP,
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

    /// Insert a memory into the database.
    /// 
    /// **Important invariant**: `memory.embedding` must always be `None` when calling this function.
    /// Embeddings are added separately via `VectorStore::add()` which updates the embedding column.
    /// 
    /// This is because:
    /// - SQLite stores embeddings as BLOB (Vec<u8>)
    /// - PostgreSQL stores embeddings as vector(384) (pgvector type)
    /// - Binding `Option<Vec<u8>>` to a PostgreSQL vector column would fail
    pub async fn insert_memory(&self, memory: &Memory) -> Result<()> {
        match self {
            Self::Sqlite(pool) => {
                sqlx::query(
                    "INSERT INTO memories (
                        id, agent_id, user_id, session_id, memory_type, text, embedding,
                        importance, is_active, supersedes_id, source_event_ids, metadata,
                        last_accessed_at, expires_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                )
                .bind(&memory.id)
                .bind(&memory.agent_id)
                .bind(&memory.user_id)
                .bind(&memory.session_id)
                .bind(&memory.memory_type)
                .bind(&memory.text)
                .bind(&memory.embedding)
                .bind(&memory.importance)
                .bind(&memory.is_active)
                .bind(&memory.supersedes_id)
                .bind(&memory.source_event_ids)
                .bind(&memory.metadata)
                .bind(&memory.last_accessed_at)
                .bind(&memory.expires_at)
                .execute(pool)
                .await?;
            }
            Self::Postgres(pool) => {
                sqlx::query(
                    "INSERT INTO memories (
                        id, agent_id, user_id, session_id, memory_type, text, embedding,
                        importance, is_active, supersedes_id, source_event_ids, metadata,
                        last_accessed_at, expires_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
                )
                .bind(&memory.id)
                .bind(&memory.agent_id)
                .bind(&memory.user_id)
                .bind(&memory.session_id)
                .bind(&memory.memory_type)
                .bind(&memory.text)
                .bind(&memory.embedding)
                .bind(&memory.importance)
                .bind(&memory.is_active)
                .bind(&memory.supersedes_id)
                .bind(&memory.source_event_ids)
                .bind(&memory.metadata)
                .bind(&memory.last_accessed_at)
                .bind(&memory.expires_at)
                .execute(pool)
                .await?;
            }
        }
        Ok(())
    }

    pub async fn get_memory(&self, id: &str) -> Result<Option<Memory>> {
        match self {
            Self::Sqlite(pool) => {
                let row = sqlx::query("SELECT * FROM memories WHERE id = ?1")
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
                let row = sqlx::query("SELECT * FROM memories WHERE id = $1")
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

    fn sqlite_row_to_memory(row: &sqlx::sqlite::SqliteRow) -> Result<Memory> {
        let is_active: i32 = row.try_get("is_active")?;
        
        Ok(Memory {
            id: row.try_get("id")?,
            agent_id: row.try_get("agent_id")?,
            user_id: row.try_get("user_id")?,
            session_id: row.try_get("session_id")?,
            memory_type: row.try_get("memory_type")?,
            text: row.try_get("text")?,
            embedding: row.try_get("embedding")?,
            importance: row.try_get("importance")?,
            is_active: is_active != 0,
            supersedes_id: row.try_get("supersedes_id")?,
            source_event_ids: row.try_get("source_event_ids")?,
            metadata: row.try_get("metadata")?,
            last_accessed_at: row.try_get("last_accessed_at")?,
            created_at: row.try_get("created_at")?,
            expires_at: row.try_get("expires_at")?,
        })
    }

    fn postgres_row_to_memory(row: &sqlx::postgres::PgRow) -> Result<Memory> {
        Ok(Memory {
            id: row.try_get("id")?,
            agent_id: row.try_get("agent_id")?,
            user_id: row.try_get("user_id")?,
            session_id: row.try_get("session_id")?,
            memory_type: row.try_get("memory_type")?,
            text: row.try_get("text")?,
            embedding: row.try_get("embedding")?,
            importance: row.try_get("importance")?,
            is_active: row.try_get("is_active")?,
            supersedes_id: row.try_get("supersedes_id")?,
            source_event_ids: row.try_get("source_event_ids")?,
            metadata: row.try_get("metadata")?,
            last_accessed_at: row.try_get("last_accessed_at")?,
            created_at: row.try_get("created_at")?,
            expires_at: row.try_get("expires_at")?,
        })
    }
}

