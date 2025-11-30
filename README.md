![Memento Logo](assets/logo.png)

# Universal Agent Memory Engine

A lightweight, local-first memory service for AI agents, built with Rust for maximum performance and safety.

[Features](#features) • [Quick Start](#quick-start) • [REST API](#-rest-api) • [MCP Integration](#-mcp-integration)

---

## Features

- **Dual Storage**: SQLite (local) and PostgreSQL (hosted) support
- **Vector Search**: Semantic search using embeddings with pgvector or SQLite BLOB storage
- **Embeddings**: Local transformers or OpenAI embeddings
- **REST API**: Full-featured HTTP API for memory operations
- **MCP Server**: Model Context Protocol server for native integration
- **Type-Safe**: Built with Rust for memory safety and performance

## Quick Start

### Prerequisites

- Rust 1.70+ and Cargo
- SQLite (bundled) or PostgreSQL with pgvector extension

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd memento

# Build the project
cargo build --release

# Run the server
cargo run -- serve
```

### Configuration

Create a `.env` file (supports both `MEMENTO_*` and non-prefixed names):

```env
# Database (use MEMENTO_DATABASE_URL or DATABASE_URL)
MEMENTO_DATABASE_URL=sqlite://./memento.db
# or
DATABASE_URL=sqlite://./memento.db

# Embeddings (optional, defaults to local)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2

# Server (only for REST API)
PORT=8000
HOST=0.0.0.0
```

For PostgreSQL:

```env
MEMENTO_DATABASE_URL=postgresql://user:pass@localhost/memento
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

## 🔌 REST API

> **Note:** The REST API is optional. It's provided for non-MCP integrations such as web applications, scripts, or services that communicate over HTTP. If you're using Cursor or another MCP-compatible client, you only need the [MCP Integration](#-mcp-integration).

### 1) Store memory/event

```bash
POST /memory/store
{
  "agent_id": "cursor",
  "user_id": "u123",
  "session_id": "s456",
  "event_type": "user_msg",
  "text": "I prefer morning meetings.",
  "metadata": {"tags": ["preference"]}
}
```

### 2) Search memories (semantic)

```bash
POST /memory/search
{
  "agent_id": "cursor",
  "user_id": "u123",
  "query": "scheduling preferences",
  "k": 5,
  "filters": {"memory_type": ["preference","semantic"]}
}
```

### 3) Summarize recent events

```bash
POST /memory/summarize
{
  "agent_id": "cursor",
  "user_id": "u123",
  "session_id": "s456"
}
```

### 4) Forget memory

```bash
POST /memory/forget
{
  "agent_id": "cursor",
  "user_id": "u123",
  "query": "phone number"
}
```

## 🤝 MCP Integration

### Quick Setup

1. **Build the binary** (one-time setup):

   ```bash
   cargo build --release
   ```

   The binary will be at `target/release/memento` (or `target/release/memento.exe` on Windows).

2. **Configure in Cursor** - Add to `~/.cursor/mcp.json` (see examples below)

3. **Done!** The MCP server will start automatically when Cursor launches.

### Tools

- `memento.store(text, agent_id, user_id?, session_id?, event_type?, metadata?)`
- `memento.search(query, agent_id, user_id?, k?, filters?)`
- `memento.summarize(agent_id, user_id?, session_id?)`
- `memento.forget(memory_id|query, agent_id, user_id?)`

### Cursor Configuration

Add to `~/.cursor/mcp.json`:

#### SQLite Configuration (Default)

**Simplest setup** - Uses defaults (SQLite database at `./memento.db` in the binary's directory):

```json
{
  "mcpServers": {
    "memento": {
      "command": "/absolute/path/to/memento/target/release/memento",
      "args": ["mcp"]
    }
  }
}
```

**Custom database location** - Specify your own database path:

```json
{
  "mcpServers": {
    "memento": {
      "command": "/absolute/path/to/memento/target/release/memento",
      "args": ["mcp"],
      "env": {
        "MEMENTO_DATABASE_URL": "/home/user/.memento/memories.db"
      }
    }
  }
}
```

**Or hardcode in args:**

```json
{
  "mcpServers": {
    "memento": {
      "command": "/absolute/path/to/memento/target/release/memento",
      "args": [
        "mcp",
        "--database-url",
        "/home/user/.memento/memories.db"
      ]
    }
  }
}
```

#### PostgreSQL Configuration

**Using environment variables (recommended):**

```json
{
  "mcpServers": {
    "memento": {
      "command": "/absolute/path/to/memento/target/release/memento",
      "args": ["mcp"],
      "env": {
        "MEMENTO_DATABASE_TYPE": "postgresql",
        "MEMENTO_DATABASE_URL": "postgresql://user:password@localhost:5432/memento"
      }
    }
  }
}
```

**Or hardcode in args:**

```json
{
  "mcpServers": {
    "memento": {
      "command": "/absolute/path/to/memento/target/release/memento",
      "args": [
        "mcp",
        "--database-type",
        "postgresql",
        "--database-url",
        "postgresql://user:password@localhost:5432/memento"
      ]
    }
  }
}
```

### How It Works

Once configured, the MCP server will:

1. **Start automatically** when Cursor launches
2. **Connect to your database** using the settings you provided
3. **Make tools available** to AI assistants for storing and retrieving memories

No additional setup or manual steps required!

### Configuration Options

**Defaults (no config needed):**

- Database type: SQLite
- Database location: `./memento.db` (relative to where the binary runs)

**To customize:**

- Set `MEMENTO_DATABASE_TYPE` and `MEMENTO_DATABASE_URL` in the `env` object, OR
- Use `--database-type` and `--database-url` in the `args` array

**Configuration Priority** (highest to lowest):

1. CLI arguments (`--database-type`, `--database-url`)
2. Environment variables (`MEMENTO_DATABASE_TYPE`, `MEMENTO_DATABASE_URL`, or `DATABASE_URL`)
3. Defaults (SQLite at `./memento.db`)

#### Environment Variables

**Database:**

- `MEMENTO_DATABASE_TYPE`: `sqlite` or `postgresql` (default: `sqlite`) - MCP only
- `MEMENTO_DATABASE_URL` or `DATABASE_URL`:
  - For SQLite: Path to database file (e.g., `./memento.db` or `/path/to/db.sqlite`)
  - For PostgreSQL: Full connection string (e.g., `postgresql://user:pass@host:5432/dbname`)
  - Default: `sqlite://./memento.db` (for both REST API and MCP)

**Embeddings (optional, for semantic search):**

- `MEMENTO_EMBEDDING_PROVIDER` or `EMBEDDING_PROVIDER`: `local` or `openai` (default: `local`)
- `MEMENTO_EMBEDDING_MODEL` or `EMBEDDING_MODEL`: Model name (default: `Xenova/all-MiniLM-L6-v2`)
- `MEMENTO_OPENAI_API_KEY` or `OPENAI_API_KEY`: Required if using OpenAI embeddings

**Server (REST API only):**

- `MEMENTO_PORT` or `PORT`: Server port (default: `8000`)
- `MEMENTO_HOST` or `HOST`: Server host (default: `0.0.0.0`)

**Note:** All environment variables support both `MEMENTO_*` prefixed and non-prefixed versions. Prefixed versions take priority.

## 🏗️ Architecture

- **Database Layer**: Abstracted over SQLite (rusqlite) and PostgreSQL (sqlx)
- **Vector Store**: Supports pgvector for PostgreSQL and BLOB storage for SQLite
- **Embeddings**: Trait-based design for local and OpenAI providers
- **REST API**: Built with Axum for async HTTP handling
- **MCP Server**: JSON-RPC over stdio for MCP protocol

## 🚀 Performance

Rust provides:

- Zero-cost abstractions
- Memory safety without garbage collection
- Concurrent request handling with Tokio
- Efficient database access with connection pooling

## 📦 Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt

# Lint code
cargo clippy
```

## 🔧 Development

The project structure:

```text
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library root
├── config.rs        # Configuration management
├── database.rs      # Database abstraction
├── embeddings.rs    # Embedding providers
├── vector_store.rs  # Vector search implementation
├── server.rs        # REST API server
├── mcp.rs           # MCP server
└── types.rs         # Type definitions
```

## 📝 License

MIT
