![Memento Logo](assets/logo.png)

# Universal Agent Memory Engine

A lightweight, local-first memory service for AI agents, built with Rust for maximum performance and safety.

[Features](#features) • [Quick Start](#quick-start) • [MCP Tools](#-mcp-tools) • [Configuration](#configuration)

---

## Features

- **MCP Native**: Model Context Protocol server for seamless AI integration
- **Dual Storage**: SQLite (local) and PostgreSQL (hosted) support
- **Vector Search**: Semantic search using embeddings (pgvector or SQLite BLOB)
- **Embeddings**: Local transformers or OpenAI embeddings
- **LLM-Delegated Summarization**: Uses your existing LLM to consolidate memories
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

# Interactive setup (optional)
./target/release/memento init
```

### Configure in Cursor

Add to `~/.cursor/mcp.json`:

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

**Done!** Restart Cursor and the memory tools are available.

## 🛠 MCP Tools

### `memento.store`

**Proactively** store important information for future recall.

```json
{
  "text": "User prefers 4-space indentation in Python",
  "agent_id": "cursor",
  "event_type": "preference",
  "metadata": {"tags": ["python", "formatting"]}
}
```

### `memento.search`

Search stored memories when asked to recall something.

```json
{
  "query": "Python formatting preferences",
  "agent_id": "cursor",
  "k": 5
}
```

### `memento.summarize`

Get unsummarized events for consolidation. Returns events with instructions for the LLM to process and store summaries.

```json
{
  "agent_id": "cursor",
  "limit": 50
}
```

### `memento.mark_summarized`

Mark events as processed after storing summaries.

```json
{
  "agent_id": "cursor",
  "event_ids": ["evt-1", "evt-2", "evt-3"]
}
```

### `memento.forget`

Remove memories by ID or query.

```json
{
  "agent_id": "cursor",
  "query": "phone number"
}
```

## Configuration

### Environment Variables

**Database:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMENTO_DATABASE_URL` | `sqlite://./memento.db` | Database connection URL |
| `MEMENTO_DATABASE_TYPE` | `sqlite` | `sqlite` or `postgresql` |

**Embeddings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMENTO_EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `MEMENTO_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | Model name |
| `MEMENTO_OPENAI_API_KEY` | - | Required for OpenAI embeddings |

### Cursor Configuration Examples

#### SQLite (Default)

```json
{
  "mcpServers": {
    "memento": {
      "command": "/path/to/memento",
      "args": ["mcp"]
    }
  }
}
```

#### Custom Database Location

```json
{
  "mcpServers": {
    "memento": {
      "command": "/path/to/memento",
      "args": ["mcp"],
      "env": {
        "MEMENTO_DATABASE_URL": "/home/user/.memento/memories.db"
      }
    }
  }
}
```

#### PostgreSQL with OpenAI Embeddings

```json
{
  "mcpServers": {
    "memento": {
      "command": "/path/to/memento",
      "args": ["mcp"],
      "env": {
        "MEMENTO_DATABASE_TYPE": "postgresql",
        "MEMENTO_DATABASE_URL": "postgresql://user:pass@localhost:5432/memento",
        "MEMENTO_EMBEDDING_PROVIDER": "openai",
        "MEMENTO_OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## 🏗️ Architecture

```text
src/
├── main.rs          # CLI entry point
├── lib.rs           # Library root
├── config.rs        # Configuration management
├── database.rs      # Database abstraction (SQLite/PostgreSQL)
├── embeddings.rs    # Embedding providers (local/OpenAI)
├── vector_store.rs  # Vector search implementation
├── mcp.rs           # MCP server (JSON-RPC over stdio)
├── bootstrap.rs     # Initialization helpers
└── types.rs         # Type definitions
```

## 📦 Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test
```

## 📝 License

MIT
