use crate::types::Metadata;
use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, Row};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Pool, Postgres, Row as SqlxRow};
use std::sync::Arc;
use uuid::Uuid;

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

pub enum DatabaseClient {
    Sqlite(Arc<Connection>),
    Postgres(PgPool),
}

impl DatabaseClient {
    pub async fn new(database_url: &str) -> Result<Self> {
        if database_url.starts_with("postgresql://") {
            let pool = PgPool::connect(database_url).await?;
            Self::init_postgres(&pool).await?;
            Ok(Self::Postgres(pool))
        } else {
            let db_path = database_url.replace("sqlite://", "");
            let conn = Connection::open(db_path)?;
            Self::init_sqlite(&conn)?;
            Ok(Self::Sqlite(Arc::new(conn)))
        }
    }

    fn init_sqlite(conn: &Connection) -> Result<()> {
        conn.execute("PRAGMA foreign_keys = ON", [])?;

        conn.execute(
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
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_agent_id ON memory_events(agent_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_user_id ON memory_events(user_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_session_id ON memory_events(session_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_event_type ON memory_events(event_type)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_created_at ON memory_events(created_at)",
            [],
        )?;

        conn.execute(
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
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
            [],
        )?;

        Ok(())
    }

    async fn init_postgres(pool: &PgPool) -> Result<()> {
        // Enable pgvector extension if needed (will be handled by vector_store)
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(pool)
            .await
            .ok(); // Ignore if already exists or not available

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
                embedding BYTEA,
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

    pub async fn insert_event(&self, event: &MemoryEvent) -> Result<()> {
        match self {
            Self::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                tokio::task::spawn_blocking(move || {
                    conn.execute(
                        "INSERT INTO memory_events (id, agent_id, user_id, session_id, event_type, content, metadata)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            event.id,
                            event.agent_id,
                            event.user_id,
                            event.session_id,
                            event.event_type,
                            event.content,
                            event.metadata
                        ],
                    )?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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

    pub async fn insert_memory(&self, memory: &Memory) -> Result<()> {
        match self {
            Self::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                let memory_clone = memory.clone();
                tokio::task::spawn_blocking(move || {
                    conn.execute(
                        "INSERT INTO memories (
                            id, agent_id, user_id, session_id, memory_type, text, embedding,
                            importance, is_active, supersedes_id, source_event_ids, metadata,
                            last_accessed_at, expires_at
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                        params![
                            memory_clone.id,
                            memory_clone.agent_id,
                            memory_clone.user_id,
                            memory_clone.session_id,
                            memory_clone.memory_type,
                            memory_clone.text,
                            memory_clone.embedding,
                            memory_clone.importance,
                            memory_clone.is_active,
                            memory_clone.supersedes_id,
                            memory_clone.source_event_ids,
                            memory_clone.metadata,
                            memory_clone.last_accessed_at,
                            memory_clone.expires_at
                        ],
                    )?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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
            Self::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                let id = id.to_string();
                tokio::task::spawn_blocking(move || {
                    let mut stmt = conn.prepare("SELECT * FROM memories WHERE id = ?1")?;
                    let memory = stmt.query_row([&id], |row| Self::row_to_memory(row, false))?;
                    Ok::<Option<Memory>, rusqlite::Error>(Some(memory))
                })
                .await?
                .map_err(Into::into)
            }
            Self::Postgres(pool) => {
                let row = sqlx::query("SELECT * FROM memories WHERE id = $1")
                    .bind(id)
                    .fetch_optional(pool)
                    .await?;
                
                if let Some(row) = row {
                    Ok(Some(Self::sqlx_row_to_memory(&row)?))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub async fn update_memory_access(&self, id: &str) -> Result<()> {
        match self {
            Self::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                let id = id.to_string();
                tokio::task::spawn_blocking(move || {
                    conn.execute(
                        "UPDATE memories SET last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?1",
                        [&id],
                    )?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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
            Self::Sqlite(conn) => {
                let conn = Arc::clone(conn);
                let id = id.to_string();
                tokio::task::spawn_blocking(move || {
                    conn.execute("UPDATE memories SET is_active = 0 WHERE id = ?1", [&id])?;
                    Ok::<(), rusqlite::Error>(())
                })
                .await??;
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

    fn row_to_memory(row: &Row, is_postgres: bool) -> rusqlite::Result<Memory> {
        Ok(Memory {
            id: row.get(0)?,
            agent_id: row.get(1)?,
            user_id: row.get(2)?,
            session_id: row.get(3)?,
            memory_type: row.get(4)?,
            text: row.get(5)?,
            embedding: row.get(6)?,
            importance: row.get(7)?,
            is_active: if is_postgres {
                row.get(8)?
            } else {
                row.get::<_, i32>(8)? != 0
            },
            supersedes_id: row.get(9)?,
            source_event_ids: row.get(10)?,
            metadata: row.get(11)?,
            last_accessed_at: row.get(12)?,
            created_at: row.get(13)?,
            expires_at: row.get(14)?,
        })
    }

    fn sqlx_row_to_memory(row: &sqlx::postgres::PgRow) -> Result<Memory> {
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

