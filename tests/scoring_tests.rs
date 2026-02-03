//! Tests for the scoring module.
//!
//! Tests importance scoring functions including:
//! - Content heuristics (text analysis)
//! - Novelty adjustment (similarity-based)
//! - Access boost (diminishing returns)

use memento::config::{
    HIGH_SIMILARITY_THRESHOLD, LOW_SIMILARITY_THRESHOLD, MAX_CONTENT_ADJUSTMENT,
    MAX_NOVELTY_ADJUSTMENT,
};
use memento::scoring::{
    boost_on_access, content_heuristics, novelty_adjustment, ACCESS_BOOST, MAX_IMPORTANCE,
};
use memento::types::Metadata;
use memento::vector_store::VectorSearchResult;

// =============================================================================
// Content Heuristics Tests
// =============================================================================

#[test]
fn test_content_heuristics_action_items() {
    let text = "We decided to implement the new feature by tomorrow. This is a priority task.";
    let adjustment = content_heuristics(text);
    assert!(adjustment > 0.0, "Action items should increase importance");
}

#[test]
fn test_content_heuristics_questions() {
    let text = "What is this?";
    let adjustment = content_heuristics(text);
    assert!(adjustment < 0.0, "Short questions should decrease importance");
}

#[test]
fn test_content_heuristics_code() {
    // Text is long enough to not trigger the "short text" penalty
    let text = "Here's the implementation for the database connection module:\n```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```";
    let adjustment = content_heuristics(text);
    assert!(adjustment > 0.0, "Code should increase importance");
}

#[test]
fn test_content_heuristics_smalltalk() {
    let text = "hello, how are you doing today?";
    let adjustment = content_heuristics(text);
    assert!(adjustment < 0.0, "Smalltalk should decrease importance");
}

#[test]
fn test_content_heuristics_empty_text() {
    let adjustment = content_heuristics("");
    // Empty text has 0 words, so it gets short text penalty
    assert!(adjustment < 0.0, "Empty text should decrease importance");
    assert!(
        adjustment >= -MAX_CONTENT_ADJUSTMENT,
        "Should be clamped to min"
    );
}

#[test]
fn test_content_heuristics_whitespace_only() {
    let adjustment = content_heuristics("   \t\n   ");
    // Whitespace-only has 0 words
    assert!(
        adjustment < 0.0,
        "Whitespace-only should decrease importance"
    );
}

#[test]
fn test_content_heuristics_clamping() {
    // Text with many negative signals
    let very_negative = "hi there?";
    let adj = content_heuristics(very_negative);
    assert!(adj >= -MAX_CONTENT_ADJUSTMENT, "Should not go below -MAX");

    // Text with many positive signals
    let very_positive = "We decided this is a critical priority task by tomorrow. \
        It's important that we must do this action item. The deadline is by end of day. \
        Here's the code:\n```rust\nfn main() {}\n```\n\
        - Item one\n- Item two\n- Item three\n\
        John Smith and Jane Doe agreed on this decision.";
    let adj = content_heuristics(very_positive);
    assert!(adj <= MAX_CONTENT_ADJUSTMENT, "Should not exceed +MAX");
}

#[test]
fn test_content_heuristics_unicode() {
    // Unicode text should not crash
    let unicode = "用户偏好：使用4个空格缩进";
    let adjustment = content_heuristics(unicode);
    assert!(adjustment.is_finite(), "Unicode should produce finite result");

    // Emoji text
    let emoji = "🚀 Important decision made! 🎉 Priority task confirmed ✅";
    let adjustment = content_heuristics(emoji);
    assert!(adjustment.is_finite(), "Emoji should produce finite result");
}

// =============================================================================
// Novelty Adjustment Tests
// =============================================================================

#[test]
fn test_novelty_no_similar() {
    let adjustment = novelty_adjustment(&[]);
    assert_eq!(adjustment, MAX_NOVELTY_ADJUSTMENT);
}

#[test]
fn test_novelty_high_similarity() {
    let similar = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: 0.95,
        metadata: Metadata::new(),
    }];
    let adjustment = novelty_adjustment(&similar);
    assert!(
        adjustment < 0.0,
        "High similarity should decrease importance"
    );
}

#[test]
fn test_novelty_low_similarity() {
    let similar = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: 0.3,
        metadata: Metadata::new(),
    }];
    let adjustment = novelty_adjustment(&similar);
    assert!(adjustment > 0.0, "Low similarity should increase importance");
}

#[test]
fn test_novelty_at_threshold_boundaries() {
    // Exactly at HIGH threshold
    let at_high = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: HIGH_SIMILARITY_THRESHOLD,
        metadata: Metadata::new(),
    }];
    let adj = novelty_adjustment(&at_high);
    // At high threshold, should be slightly negative (interpolation yields ~0)
    assert!(adj <= 0.0, "At high threshold should not be positive");

    // Exactly at LOW threshold
    let at_low = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: LOW_SIMILARITY_THRESHOLD,
        metadata: Metadata::new(),
    }];
    let adj = novelty_adjustment(&at_low);
    // At low threshold, should be at max positive
    assert_eq!(
        adj, MAX_NOVELTY_ADJUSTMENT,
        "At low threshold should be max positive"
    );

    // Midpoint between thresholds
    let midpoint = (HIGH_SIMILARITY_THRESHOLD + LOW_SIMILARITY_THRESHOLD) / 2.0;
    let at_mid = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: midpoint,
        metadata: Metadata::new(),
    }];
    let adj = novelty_adjustment(&at_mid);
    // Midpoint should yield ~0 adjustment
    assert!(
        adj.abs() < 0.01,
        "Midpoint should yield near-zero adjustment"
    );
}

#[test]
fn test_novelty_nan_handling() {
    // All NaN scores - should treat as novel
    let all_nan = vec![
        VectorSearchResult {
            memory_id: "test1".to_string(),
            score: f64::NAN,
            metadata: Metadata::new(),
        },
        VectorSearchResult {
            memory_id: "test2".to_string(),
            score: f64::NAN,
            metadata: Metadata::new(),
        },
    ];
    let adj = novelty_adjustment(&all_nan);
    assert_eq!(
        adj, MAX_NOVELTY_ADJUSTMENT,
        "All NaN should be treated as novel"
    );

    // Mixed NaN and valid scores - should use valid score
    let mixed = vec![
        VectorSearchResult {
            memory_id: "test1".to_string(),
            score: f64::NAN,
            metadata: Metadata::new(),
        },
        VectorSearchResult {
            memory_id: "test2".to_string(),
            score: 0.95, // High similarity
            metadata: Metadata::new(),
        },
    ];
    let adj = novelty_adjustment(&mixed);
    assert!(adj < 0.0, "Valid high score should still reduce importance");
}

#[test]
fn test_novelty_infinity_handling() {
    // Infinity scores should be filtered
    let with_inf = vec![
        VectorSearchResult {
            memory_id: "test1".to_string(),
            score: f64::INFINITY,
            metadata: Metadata::new(),
        },
        VectorSearchResult {
            memory_id: "test2".to_string(),
            score: 0.6,
            metadata: Metadata::new(),
        },
    ];
    let adj = novelty_adjustment(&with_inf);
    // Should use the 0.6 score, not infinity
    assert!(adj.is_finite(), "Result should be finite");
}

#[test]
fn test_novelty_out_of_range_scores() {
    // Score > 1.0 should be clamped
    let too_high = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: 1.5,
        metadata: Metadata::new(),
    }];
    let adj = novelty_adjustment(&too_high);
    assert_eq!(
        adj, -MAX_NOVELTY_ADJUSTMENT,
        "Score > 1.0 should be treated as redundant"
    );

    // Score < 0.0 should be clamped
    let negative = vec![VectorSearchResult {
        memory_id: "test".to_string(),
        score: -0.5,
        metadata: Metadata::new(),
    }];
    let adj = novelty_adjustment(&negative);
    assert_eq!(
        adj, MAX_NOVELTY_ADJUSTMENT,
        "Negative score should be treated as novel"
    );
}

// =============================================================================
// Access Boost Tests
// =============================================================================

#[test]
fn test_boost_on_access() {
    let boosted = boost_on_access(0.5);
    assert!(boosted > 0.5, "Access should boost importance");
    assert!(boosted < 0.6, "Boost should be modest");
}

#[test]
fn test_boost_diminishing_returns() {
    let low_boost = boost_on_access(0.3);
    let high_boost = boost_on_access(0.9);
    let low_delta = low_boost - 0.3;
    let high_delta = high_boost - 0.9;
    assert!(
        low_delta > high_delta,
        "Boost should diminish at higher importance"
    );
}

#[test]
fn test_boost_caps_at_max() {
    let boosted = boost_on_access(0.99);
    assert!(
        boosted <= MAX_IMPORTANCE,
        "Importance should not exceed MAX"
    );
}

#[test]
fn test_boost_at_zero() {
    let boosted = boost_on_access(0.0);
    assert!(boosted > 0.0, "Zero importance should still get boosted");
    assert_eq!(boosted, ACCESS_BOOST, "Full boost at zero importance");
}

#[test]
fn test_boost_at_max() {
    let boosted = boost_on_access(MAX_IMPORTANCE);
    assert_eq!(
        boosted, MAX_IMPORTANCE,
        "At max importance, no further boost"
    );
}
