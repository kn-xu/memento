# 🏗️ Memento Architecture Overview

## 📋 Table of Contents
1. [Application Flow](#application-flow)
2. [File-by-File Breakdown](#file-by-file-breakdown)
3. [Data Flow Examples](#data-flow-examples)
4. [Architecture Patterns](#architecture-patterns)
5. [Current Limitations / TODOs](#current-limitations--todos)
6. [Summary](#summary)

---

## 🔄 Application Flow

```
User Request → CLI (main.rs) → Server/MCP → Database/VectorStore → Response
```

---

## 📁 File-by-File Breakdown

### 1. `Cargo.toml` — Project Configuration

**Purpose**: Defines all dependencies and project metadata.

**Key Dependencies**:
- **`axum`** — Modern HTTP server framework (async, type-safe routing)
- **`tokio`** — Async runtime for Rust
- **`sqlx` + `rusqlite`** — Database drivers (PostgreSQL + SQLite)
- **`pgvector`** — Vector search extension for PostgreSQL
- **`reqwest`** — HTTP client (for OpenAI embeddings)
- **`serde`** — Serialization/deserialization framework
- **`clap`** — Command-line argument parsing

---

### 2. `src/main.rs` — Entry Point

**Purpose**: CLI entry point that routes to either REST API or MCP server.

**How it works**:
1. Loads environment variables from `.env` file
2. Parses CLI arguments using `clap`:
   - `cargo run -- serve` → Starts REST API
   - `cargo run -- mcp` → Starts MCP server (stdio)
   - Default → Starts REST API on port 8000
3. Creates database connection
4. Starts the selected server

**Key Code**:
```rust
match cli.command {
    Some(Commands::Serve { port, host }) => {
        let db = DatabaseClient::new(&config.database_url).await?;
        start_server(db, host, port).await?;
    }
    Some(Commands::Mcp) => {
        memento::mcp::start_mcp_server().await?;
    }
}
```

---

### 3. `src/lib.rs` — Library Root

**Purpose**: Exposes all modules and re-exports common types.

**What it does**:
- Declares all modules (`config`, `database`, `embeddings`, etc.)
- Re-exports commonly used types (`Config`, `DatabaseClient`, `Memory`, `MemoryEvent`)

---

### 4. `src/config.rs` — Configuration Management

**Purpose**: Loads and validates configuration from environment variables.

**Key Structures**:
- **`Config`** — Main configuration struct with all settings
- **`VectorStoreType`** — Enum: `SqliteVss`, `Pgvector`, `Chroma`
- **`EmbeddingProvider`** — Enum: `Local`, `OpenAi`
- **`McpTransport`** — Enum: `Stdio`, `Sse`

**How it works**:
- Reads from environment variables with sensible defaults
- Converts strings to typed enums
- Example: `DATABASE_URL=sqlite://./memento.db` → `Config.database_url`

**Environment Variables**:
- `DATABASE_URL` — Database connection string
- `VECTOR_STORE` — Vector store type (sqlite-vss, pgvector, chroma)
- `EMBEDDING_PROVIDER` — Embedding provider (local, openai)
- `OPENAI_API_KEY` — OpenAI API key (if using OpenAI)
- `EMBEDDING_MODEL` — Model name for embeddings
- `PORT` — Server port (default: 8000)
- `HOST` — Server host (default: 0.0.0.0)

---

### 5. `src/types.rs` — Type Definitions

**Purpose**: Defines all request/response types for REST API and MCP.

**Key Types**:
- **Request Types**: `StoreRequest`, `SearchRequest`, `SummarizeRequest`, `ForgetRequest`
- **Response Types**: `StoreResponse`, `SearchResponse`, `SearchResult`, etc.
- **MCP Tool Args**: `StoreToolArgs`, `SearchToolArgs`, etc.
- **`Metadata`** — Type alias for `HashMap<String, serde_json::Value>`

**Why it matters**: Centralizes all API contract types, ensuring consistency between REST and MCP interfaces.

---

### 6. `src/database.rs` — Database Abstraction Layer

**Purpose**: Provides a unified interface for both SQLite and PostgreSQL.

**Key Structures**:
- **`MemoryEvent`** — Raw event log entry (episodic log)
- **`Memory`** — Derived memory with embedding (durable, searchable)
- **`DatabaseClient`** — Enum that wraps either SQLite or PostgreSQL connection

**How it works**:

#### 1. Initialization
- Detects database type from URL (`postgresql://` vs `sqlite://`)
- Creates tables and indexes if they don't exist
- Enables pgvector extension for PostgreSQL

#### 2. Operations
- **`insert_event()`** — Stores raw events in `memory_events` table
- **`insert_memory()`** — Stores memories with embeddings in `memories` table
- **`get_memory()`** — Retrieves memory by ID
- **`update_memory_access()`** — Updates `last_accessed_at` timestamp
- **`soft_delete_memory()`** — Sets `is_active = false` (soft delete)

#### 3. Async Handling
- **SQLite**: Uses `tokio::task::spawn_blocking` (blocking I/O in background thread)
- **PostgreSQL**: Native async with `sqlx`

**Database Schema**:

```sql
memory_events:
  - id (TEXT PRIMARY KEY)
  - agent_id (TEXT NOT NULL)
  - user_id (TEXT)
  - session_id (TEXT)
  - event_type (TEXT NOT NULL)
  - content (TEXT NOT NULL)
  - metadata (TEXT)
  - created_at (TIMESTAMP)

memories:
  - id (TEXT PRIMARY KEY)
  - agent_id (TEXT NOT NULL)
  - user_id (TEXT)
  - session_id (TEXT)
  - memory_type (TEXT NOT NULL)
  - text (TEXT NOT NULL)
  - embedding (BLOB/vector)
  - importance (REAL)
  - is_active (BOOLEAN)
  - supersedes_id (TEXT)
  - source_event_ids (TEXT)
  - metadata (TEXT)
  - last_accessed_at (TIMESTAMP)
  - created_at (TIMESTAMP)
  - expires_at (TIMESTAMP)
```

---

### 7. `src/embeddings.rs` — Embedding Providers

**Purpose**: Trait-based embedding system supporting local and OpenAI providers.

**Key Components**:
- **`EmbeddingProvider` trait** — Defines `embed()` and `embed_batch()` methods
- **`LocalEmbeddingProvider`** — Placeholder implementation (needs actual model loading)
- **`OpenAIEmbeddingProvider`** — Calls OpenAI API

**How it works**:

#### 1. Local Provider (Placeholder)
```rust
// Currently returns dummy embeddings
// TODO: Load actual transformer model (e.g., using candle-transformers)
```

#### 2. OpenAI Provider
- Makes HTTP POST to `https://api.openai.com/v1/embeddings`
- Extracts embedding vector from JSON response
- Returns `Vec<f32>` (embedding vector)

#### 3. Factory Function
```rust
get_embedding_provider("openai", api_key, model) → OpenAIEmbeddingProvider
get_embedding_provider("local", None, model) → LocalEmbeddingProvider
```

---

### 8. `src/vector_store.rs` — Vector Search Engine

**Purpose**: Manages vector embeddings and performs semantic search.

**Key Structures**:
- **`VectorStore`** — Main struct containing database and embedding provider
- **`VectorSearchResult`** — Search result with score and metadata

**How it works**:

#### 1. Adding Embeddings (`add()`)
- Generates embedding if not provided
- **PostgreSQL**: Stores as `pgvector::Vector` type
- **SQLite**: Stores as BLOB (little-endian f32 bytes)

#### 2. Searching (`search()`)
- Generates query embedding
- Routes to PostgreSQL or SQLite implementation

#### 3. PostgreSQL Search (`search_pgvector()`)
- Uses `<=>` operator (cosine distance)
- Query: `1 - (embedding <=> $1::vector) as score`
- Orders by distance, limits to `k` results

#### 4. SQLite Search (`search_sqlite()`)
- Fetches all matching memories
- Converts BLOB back to `Vec<f32>`
- Computes cosine similarity in memory
- Sorts and truncates to top `k`

#### 5. Cosine Similarity
```rust
dot_product / (norm_a.sqrt() * norm_b.sqrt())
```

---

### 9. `src/server.rs` — REST API Server

**Purpose**: HTTP API built with Axum framework.

**Key Components**:
- **`AppState`** — Shared state (database + vector store)
- **Route Handlers**: `store_memory`, `search_memory`, `summarize_memory`, `forget_memory`

**How it works**:

#### 1. Server Setup
```rust
Router::new()
    .route("/health", get(health))
    .route("/memory/store", post(store_memory))
    .route("/memory/search", post(search_memory))
    // ...
    .with_state(app_state)
```

#### 2. Store Endpoint (`store_memory`)
- Validates `agent_id` and `text`
- Creates `MemoryEvent` and stores it
- If text > 10 chars and event_type is `user_msg` or `thought`:
  - Creates `Memory` record
  - Generates embedding via `vector_store.add()`
- Returns `event_id` and optional `memory_id`

#### 3. Search Endpoint (`search_memory`)
- Calls `vector_store.search()` with query
- Fetches full memory records from database
- Updates `last_accessed_at`
- Merges metadata and returns results

#### 4. Forget Endpoint (`forget_memory`)
- If `memory_id` provided: soft-deletes that memory
- If `query` provided: searches, then soft-deletes matches
- Removes embeddings from vector store

#### 5. Error Handling
- Returns `(StatusCode, Json<ErrorResponse>)` tuples
- Converts `anyhow::Result` to HTTP errors

**API Endpoints**:
- `GET /health` — Health check
- `POST /memory/store` — Store memory/event
- `POST /memory/search` — Semantic search
- `POST /memory/summarize` — Summarize events (TODO)
- `POST /memory/forget` — Delete memories

---

### 10. `src/mcp.rs` — MCP Server

**Purpose**: Model Context Protocol server over stdio.

**How it works**:

#### 1. Server Loop
```rust
for line in reader.lines() {
    let request: Value = serde_json::from_str(&line)?;
    let response = handle_mcp_request(&request, &db, &vector_store).await?;
    serde_json::to_writer(&mut stdout, &response)?;
}
```

#### 2. Request Handling
- **`tools/list`** — Returns available tools (store, search, summarize, forget)
- **`tools/call`** — Executes tool based on name
  - Currently returns placeholder responses
  - Should mirror REST API logic

#### 3. Communication
- Reads JSON-RPC from stdin (line-delimited)
- Writes JSON-RPC to stdout
- Logs to stderr

**MCP Tools**:
- `memento.store` — Store memory/event
- `memento.search` — Semantic search
- `memento.summarize` — Summarize events
- `memento.forget` — Delete memories

---

## 🔄 Data Flow Examples

### Example 1: Storing a Memory

```
1. POST /memory/store
   ↓
2. server.rs::store_memory()
   ↓
3. database.rs::insert_event() → Stores raw event
   ↓
4. database.rs::insert_memory() → Creates memory record
   ↓
5. embeddings.rs::embed() → Generates embedding vector
   ↓
6. vector_store.rs::add() → Stores embedding in database
   ↓
7. Returns {event_id, memory_id}
```

### Example 2: Searching Memories

```
1. POST /memory/search
   ↓
2. server.rs::search_memory()
   ↓
3. embeddings.rs::embed() → Embed query text
   ↓
4. vector_store.rs::search() → Find similar vectors
   ↓
5. database.rs::get_memory() → Fetch full memory records
   ↓
6. database.rs::update_memory_access() → Update timestamp
   ↓
7. Returns {results: [{memory_id, text, score, metadata}]}
```

### Example 3: Forgetting a Memory

```
1. POST /memory/forget
   ↓
2. server.rs::forget_memory()
   ↓
3. If memory_id:
   - database.rs::soft_delete_memory()
   - vector_store.rs::delete()
   ↓
4. If query:
   - vector_store.rs::search() → Find matches
   - database.rs::soft_delete_memory() → Delete each
   - vector_store.rs::delete() → Remove embeddings
   ↓
5. Returns {deleted: count}
```

---

## 🎯 Architecture Patterns

### 1. Enum-Based Polymorphism
- `DatabaseClient` enum abstracts SQLite/PostgreSQL differences
- Pattern matching handles database-specific logic

### 2. Trait-Based Design
- `EmbeddingProvider` trait allows pluggable embedding systems
- Easy to add new providers (e.g., Cohere, HuggingFace)

### 3. Async/Await
- Uses Tokio for concurrent operations
- SQLite operations run in blocking thread pool
- PostgreSQL operations are native async

### 4. Type Safety
- Strong typing throughout (no `any` types)
- Request/response types ensure API contract compliance

### 5. Error Handling
- `anyhow::Result` for error propagation
- HTTP errors converted to proper status codes

---

## ⚠️ Current Limitations / TODOs

### 1. Local Embeddings
- **Status**: Placeholder implementation
- **TODO**: Load actual transformer model (e.g., using `candle-transformers`)
- **Current**: Returns dummy embeddings based on text hash

### 2. MCP Server
- **Status**: Tool implementations are placeholders
- **TODO**: Implement actual tool logic (mirror REST API)
- **Current**: Returns "implemented" messages

### 3. Vector Search Filtering
- **Status**: Simplified implementation
- **TODO**: Use proper query builder for dynamic filters
- **Current**: Basic filtering, some done in memory

### 4. Summarization
- **Status**: Not implemented
- **TODO**: Implement LLM-based summarization worker
- **Current**: Returns placeholder response

### 5. SQLite Vector Search
- **Status**: BLOB storage with in-memory similarity
- **TODO**: Integrate `sqlite-vss` extension for native vector search
- **Current**: Works but not optimal for large datasets

---

## 📝 Summary

**Memento** is a memory engine for AI agents that:

✅ **Stores** events and memories in SQLite or PostgreSQL  
✅ **Generates** embeddings using OpenAI or local models  
✅ **Searches** semantically using vector similarity  
✅ **Exposes** REST API and MCP server interfaces  
✅ **Handles** multi-tenant data (agent_id, user_id, session_id)  

The architecture is:
- **Modular** — Clear separation of concerns
- **Type-Safe** — Strong typing throughout
- **Performant** — Rust's zero-cost abstractions
- **Extensible** — Easy to add new providers/features

---

## 🚀 Quick Reference

### Running the Server
```bash
# REST API (default)
cargo run -- serve

# REST API (custom port)
cargo run -- serve --port 3000

# MCP Server
cargo run -- mcp
```

### Environment Variables
```bash
DATABASE_URL=sqlite://./memento.db
VECTOR_STORE=sqlite-vss
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2
PORT=8000
HOST=0.0.0.0
```

### API Examples
```bash
# Store memory
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "test", "text": "I like pizza"}'

# Search memories
curl -X POST http://localhost:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "test", "query": "food preferences", "k": 5}'
```

---

**Last Updated**: 2024  
**Version**: 0.1.0

