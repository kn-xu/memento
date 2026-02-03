//! Configuration constants and runtime settings.
//!
//! This module contains:
//! - Compile-time constants (scoring parameters, limits)
//! - Runtime configuration (loaded from environment variables)

use serde::{Deserialize, Serialize};
use std::env;

// =============================================================================
// Input Validation Limits
// =============================================================================

/// Maximum text length allowed for a single memory (100KB).
pub const MAX_TEXT_LENGTH: usize = 100 * 1024;

/// Maximum metadata JSON size allowed (64KB).
pub const MAX_METADATA_SIZE: usize = 64 * 1024;

/// Maximum number of results allowed per search query.
pub const MAX_SEARCH_RESULTS: usize = 100;

// =============================================================================
// Importance Scoring Parameters
// =============================================================================

/// Base importance score for all memories.
pub const BASE_IMPORTANCE: f64 = 0.5;

/// Maximum adjustment from content heuristics.
pub const MAX_CONTENT_ADJUSTMENT: f64 = 0.2;

/// Maximum adjustment from novelty scoring.
pub const MAX_NOVELTY_ADJUSTMENT: f64 = 0.2;

/// Similarity threshold above which memory is considered redundant.
pub const HIGH_SIMILARITY_THRESHOLD: f64 = 0.85;

/// Similarity threshold below which memory is considered novel.
pub const LOW_SIMILARITY_THRESHOLD: f64 = 0.5;

/// Precomputed range between similarity thresholds.
pub const SIMILARITY_THRESHOLD_RANGE: f64 = HIGH_SIMILARITY_THRESHOLD - LOW_SIMILARITY_THRESHOLD;

/// Boost applied when a memory is accessed/retrieved.
pub const ACCESS_BOOST: f64 = 0.02;

/// Maximum importance value (cap for access boosts).
pub const MAX_IMPORTANCE: f64 = 1.0;

// =============================================================================
// Runtime Configuration
// =============================================================================

/// Runtime configuration loaded from environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub embedding_provider: EmbeddingProvider,
    pub openai_api_key: Option<String>,
    pub embedding_model: String,
    pub embedding_dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "openai")]
    OpenAi,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        Self::Local
    }
}

impl From<String> for EmbeddingProvider {
    fn from(s: String) -> Self {
        match s.as_str() {
            "openai" => Self::OpenAi,
            _ => Self::Local,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("MEMENTO_DATABASE_URL")
                .unwrap_or_else(|_| "sqlite://./db/memento.db".to_string()),
            embedding_provider: env::var("MEMENTO_EMBEDDING_PROVIDER")
                .unwrap_or_else(|_| "local".to_string())
                .into(),
            openai_api_key: env::var("MEMENTO_OPENAI_API_KEY").ok(),
            embedding_model: env::var("MEMENTO_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "Xenova/all-MiniLM-L6-v2".to_string()),
            embedding_dim: env::var("MEMENTO_EMBEDDING_DIM")
                .ok()
                .and_then(|s| s.parse().ok()),
        }
    }
}

// =============================================================================
// Compile-time Validation
// =============================================================================

const _: () = {
    // Limit validations
    assert!(MAX_TEXT_LENGTH >= 1024, "MAX_TEXT_LENGTH must be >= 1KB");
    assert!(MAX_TEXT_LENGTH <= 10 * 1024 * 1024, "MAX_TEXT_LENGTH must be <= 10MB");
    assert!(MAX_METADATA_SIZE >= 1024, "MAX_METADATA_SIZE must be >= 1KB");
    assert!(MAX_METADATA_SIZE <= 1024 * 1024, "MAX_METADATA_SIZE must be <= 1MB");
    assert!(MAX_SEARCH_RESULTS >= 1, "MAX_SEARCH_RESULTS must be >= 1");
    assert!(MAX_SEARCH_RESULTS <= 1000, "MAX_SEARCH_RESULTS must be <= 1000");

    // Scoring validations
    assert!(HIGH_SIMILARITY_THRESHOLD > LOW_SIMILARITY_THRESHOLD, 
        "HIGH_SIMILARITY_THRESHOLD must be > LOW_SIMILARITY_THRESHOLD");
    assert!(LOW_SIMILARITY_THRESHOLD >= 0.0, "LOW_SIMILARITY_THRESHOLD must be >= 0");
    assert!(HIGH_SIMILARITY_THRESHOLD <= 1.0, "HIGH_SIMILARITY_THRESHOLD must be <= 1");
    assert!(BASE_IMPORTANCE >= 0.0, "BASE_IMPORTANCE must be >= 0");
    assert!(BASE_IMPORTANCE <= MAX_IMPORTANCE, "BASE_IMPORTANCE must be <= MAX_IMPORTANCE");
    assert!(MAX_CONTENT_ADJUSTMENT >= 0.0, "MAX_CONTENT_ADJUSTMENT must be >= 0");
    assert!(MAX_NOVELTY_ADJUSTMENT >= 0.0, "MAX_NOVELTY_ADJUSTMENT must be >= 0");
};
