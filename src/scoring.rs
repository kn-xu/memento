//! Importance scoring for memories.
//!
//! Calculates memory importance using multiple signals:
//! - Content heuristics (text analysis)
//! - Embedding novelty (comparison to existing memories)
//! - User override (explicit importance value)

use crate::config::{
    BASE_IMPORTANCE, HIGH_SIMILARITY_THRESHOLD, LOW_SIMILARITY_THRESHOLD,
    MAX_CONTENT_ADJUSTMENT, MAX_NOVELTY_ADJUSTMENT, SIMILARITY_THRESHOLD_RANGE,
};
use crate::vector_store::{VectorSearchResult, VectorStore};

// Re-export constants for external use
pub use crate::config::{ACCESS_BOOST, MAX_IMPORTANCE};

// =============================================================================
// Content Heuristics
// =============================================================================

/// Analyzes text content and returns an importance adjustment.
/// Returns a value between -MAX_CONTENT_ADJUSTMENT and +MAX_CONTENT_ADJUSTMENT.
pub fn content_heuristics(text: &str) -> f64 {
    let mut adjustment = 0.0;

    // Length signals
    let word_count = text.split_whitespace().count();
    if word_count > 50 {
        adjustment += 0.05; // Substantial content
    } else if word_count < 10 {
        adjustment -= 0.05; // Very short
    }

    // Action/decision indicators (more important)
    let action_patterns = [
        "decided", "decision", "agreed", "confirmed",
        "will do", "must", "should", "need to",
        "action item", "todo", "task",
        "important", "critical", "priority",
        "deadline", "by tomorrow", "by end of",
    ];
    let lower_text = text.to_lowercase();
    let action_matches = action_patterns
        .iter()
        .filter(|p| lower_text.contains(*p))
        .count();
    adjustment += (action_matches as f64 * 0.03).min(0.1);

    // Contains structured data (code, lists, etc.)
    let has_fn_keyword = text.starts_with("fn ") || text.contains(" fn ") || text.contains("\nfn ") || text.contains("\tfn ");
    if text.contains("```") || has_fn_keyword || text.contains("function ") {
        adjustment += 0.05; // Code snippets
    }
    if text.lines().filter(|l| l.trim().starts_with('-') || l.trim().starts_with('*')).count() >= 3 {
        adjustment += 0.03; // Lists
    }

    // Contains names/entities (capitalized words mid-sentence)
    let capitalized_words = text
        .split_whitespace()
        .skip(1) // Skip first word
        .filter(|w| {
            w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                && w.len() > 1
                && !w.chars().all(|c| c.is_uppercase()) // Not all caps
        })
        .count();
    if capitalized_words >= 2 {
        adjustment += 0.03; // Contains names/entities
    }

    // Questions are often ephemeral (less important to remember)
    if text.trim().ends_with('?') && word_count < 20 {
        adjustment -= 0.08;
    }

    // Greetings and small talk (less important)
    let smalltalk_patterns = ["hello", "hi there", "how are you", "thanks", "thank you", "goodbye"];
    if smalltalk_patterns.iter().any(|p| lower_text.starts_with(p)) {
        adjustment -= 0.1;
    }

    adjustment.clamp(-MAX_CONTENT_ADJUSTMENT, MAX_CONTENT_ADJUSTMENT)
}

// =============================================================================
// Novelty Scoring
// =============================================================================

/// Calculates novelty score based on similarity to existing memories.
/// Returns a value between -MAX_NOVELTY_ADJUSTMENT and +MAX_NOVELTY_ADJUSTMENT.
///
/// - High similarity to existing memories → lower importance (redundant)
/// - Low similarity → higher importance (novel information)
///
/// # Edge Cases
/// - Empty input: returns MAX_NOVELTY_ADJUSTMENT (everything is novel)
/// - NaN scores: filtered out, treated as if not present
/// - All NaN scores: treated same as empty input
pub fn novelty_adjustment(similar_memories: &[VectorSearchResult]) -> f64 {
    if similar_memories.is_empty() {
        // No existing memories = everything is novel
        return MAX_NOVELTY_ADJUSTMENT;
    }

    // Filter out NaN scores and find maximum valid similarity
    let max_similarity = similar_memories
        .iter()
        .map(|r| r.score)
        .filter(|s| s.is_finite()) // Filter NaN and infinity
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // If no valid scores found, treat as novel
    let max_similarity = match max_similarity {
        Some(s) => s,
        None => return MAX_NOVELTY_ADJUSTMENT,
    };

    // Clamp to valid range [0.0, 1.0] for robustness
    let max_similarity = max_similarity.clamp(0.0, 1.0);

    if max_similarity > HIGH_SIMILARITY_THRESHOLD {
        // Very similar to existing memory - likely redundant
        -MAX_NOVELTY_ADJUSTMENT
    } else if max_similarity < LOW_SIMILARITY_THRESHOLD {
        // Novel information
        MAX_NOVELTY_ADJUSTMENT
    } else {
        // Linear interpolation between thresholds
        // Uses precomputed SIMILARITY_THRESHOLD_RANGE (validated non-zero at compile time)
        let position = (max_similarity - LOW_SIMILARITY_THRESHOLD) / SIMILARITY_THRESHOLD_RANGE;
        // Map 0..1 to +MAX..-MAX
        MAX_NOVELTY_ADJUSTMENT * (1.0 - 2.0 * position)
    }
}

// =============================================================================
// Main Scoring Function
// =============================================================================

/// Calculates the importance score for a memory.
///
/// # Arguments
/// * `text` - The memory text content
/// * `embedding` - Pre-computed embedding for the text
/// * `vector_store` - Vector store for novelty comparison
/// * `agent_id` - Agent ID for scoped similarity search
/// * `user_override` - Optional explicit importance value (0.0-1.0)
///
/// # Returns
/// Importance score between 0.0 and 1.0
pub async fn calculate_importance(
    text: &str,
    embedding: &[f32],
    vector_store: &VectorStore,
    agent_id: &str,
    user_id: Option<&str>,
    user_override: Option<f64>,
) -> f64 {
    // If user provides explicit importance, use it directly
    if let Some(importance) = user_override {
        return importance.clamp(0.0, MAX_IMPORTANCE);
    }

    let mut score = BASE_IMPORTANCE;

    // Apply content heuristics (fast, local analysis)
    score += content_heuristics(text);

    // Apply novelty scoring (uses existing embeddings)
    match vector_store
        .search_by_embedding(embedding, 5, agent_id, user_id)
        .await
    {
        Ok(similar) => {
            score += novelty_adjustment(&similar);
        }
        Err(e) => {
            // Log but don't fail - just skip novelty adjustment
            eprintln!("⚠️  Novelty scoring failed, using content heuristics only: {}", e);
        }
    }

    score.clamp(0.0, MAX_IMPORTANCE)
}

/// Calculates a boosted importance value after a memory is accessed.
/// Uses diminishing returns to prevent importance from growing unboundedly.
pub fn boost_on_access(current_importance: f64) -> f64 {
    // Diminishing returns: boost decreases as importance approaches MAX
    let headroom = MAX_IMPORTANCE - current_importance;
    let effective_boost = ACCESS_BOOST * (headroom / MAX_IMPORTANCE);
    (current_importance + effective_boost).min(MAX_IMPORTANCE)
}
