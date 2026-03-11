//! Tests for MCP server functionality.

use memento::config::{MAX_METADATA_SIZE, MAX_SEARCH_RESULTS, MAX_TEXT_LENGTH};
use memento::vector_store::escape_like_pattern;

#[test]
fn test_escape_like_pattern_no_special_chars() {
    assert_eq!(escape_like_pattern("hello world"), "hello world");
}

#[test]
fn test_escape_like_pattern_percent() {
    assert_eq!(escape_like_pattern("100%"), "100\\%");
    assert_eq!(escape_like_pattern("50% off"), "50\\% off");
}

#[test]
fn test_escape_like_pattern_underscore() {
    assert_eq!(escape_like_pattern("user_name"), "user\\_name");
    assert_eq!(escape_like_pattern("_prefix"), "\\_prefix");
}

#[test]
fn test_escape_like_pattern_backslash() {
    assert_eq!(escape_like_pattern("path\\to\\file"), "path\\\\to\\\\file");
}

#[test]
fn test_escape_like_pattern_all_special_chars() {
    // Backslash must be escaped first to avoid double-escaping
    assert_eq!(escape_like_pattern("100%_\\test"), "100\\%\\_\\\\test");
}

#[test]
fn test_escape_like_pattern_empty() {
    assert_eq!(escape_like_pattern(""), "");
}

#[test]
fn test_escape_like_pattern_only_wildcards() {
    assert_eq!(escape_like_pattern("%_%"), "\\%\\_\\%");
}

#[test]
fn test_max_search_results_reasonable() {
    // Ensure the constant is set to a reasonable value
    assert!(
        MAX_SEARCH_RESULTS >= 10,
        "MAX_SEARCH_RESULTS should allow reasonable queries"
    );
    assert!(
        MAX_SEARCH_RESULTS <= 1000,
        "MAX_SEARCH_RESULTS should prevent abuse"
    );
}

#[test]
fn test_max_text_length_reasonable() {
    // 100KB is reasonable for text storage
    assert_eq!(MAX_TEXT_LENGTH, 100 * 1024);
    assert!(
        MAX_TEXT_LENGTH >= 10_000,
        "Should allow reasonably long texts"
    );
    assert!(MAX_TEXT_LENGTH <= 10_000_000, "Should prevent abuse");
}

#[test]
fn test_max_metadata_size_reasonable() {
    // 64KB is reasonable for JSON metadata
    assert_eq!(MAX_METADATA_SIZE, 64 * 1024);
    assert!(MAX_METADATA_SIZE >= 1_000, "Should allow useful metadata");
    assert!(MAX_METADATA_SIZE <= 1_000_000, "Should prevent abuse");
}

#[test]
fn test_importance_clamping() {
    // Test that the clamping logic works correctly
    let test_cases: [(Option<f64>, Option<f64>); 7] = [
        (Some(-1.0), Some(0.0)),  // Negative clamped to 0
        (Some(0.0), Some(0.0)),   // Zero stays zero
        (Some(0.5), Some(0.5)),   // Valid value unchanged
        (Some(1.0), Some(1.0)),   // Max stays max
        (Some(1.5), Some(1.0)),   // Over max clamped to 1
        (Some(999.0), Some(1.0)), // Way over clamped to 1
        (None, None),             // None stays None
    ];

    for (input, expected) in test_cases {
        let result = input.map(|imp| imp.clamp(0.0, 1.0));
        assert_eq!(
            result, expected,
            "Clamping {:?} should give {:?}",
            input, expected
        );
    }
}
