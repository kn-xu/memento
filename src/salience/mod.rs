//! Salience module — emotional intensity scoring for memories.
//!
//! Scores are computed at ingestion time and modulate:
//!  - encoding priority (high-salience memories are promoted immediately)
//!  - decay rate       (emotionally intense memories decay more slowly)
//!  - retrieval rank   (salience is blended into the final similarity score)
//!
//! The structural scorer is always available. The ONNX neural scorer is gated
//! behind the `onnx` cargo feature.

use crate::database::DatabaseClient;
use crate::types::{EmotionTag, OnnxEmotion, SalienceScore, SalienceSignals};
use anyhow::Result;
use serde::{Deserialize, Serialize};

// =============================================================================
// Blending weights
// =============================================================================

const WEIGHT_STRUCTURAL: f32 = 0.45;
const WEIGHT_ONNX: f32 = 0.55;

// =============================================================================
// Word lists used by signal detectors
// =============================================================================

const URGENCY_WORDS: &[&str] = &[
    "urgent", "asap", "immediately", "critical", "emergency", "right away",
];

const CORRECTION_PHRASES: &[&str] = &[
    "no,",
    "no.",
    "that's wrong",
    "not what i meant",
    "actually,",
    "wait,",
    "wrong",
    "incorrect",
];

const FRUSTRATION_WORDS: &[&str] = &[
    "frustrated", "annoying", "useless", "terrible", "awful", "horrible",
    "broken", "keeps failing", "not working", "why won't", "hate",
];

const EXCITEMENT_WORDS: &[&str] = &[
    "amazing", "awesome", "fantastic", "excellent", "perfect", "love it",
    "incredible", "brilliant", "wow", "great", "excited",
];

const CONFUSION_WORDS: &[&str] = &[
    "confused", "don't understand", "unclear", "what does", "how do i",
    "not sure", "doesn't make sense", "lost", "stuck",
];

const SATISFACTION_WORDS: &[&str] = &[
    "thank you", "thanks", "that worked", "solved", "fixed", "got it",
    "makes sense", "perfect", "exactly what", "appreciate",
];

// =============================================================================
// Supporting types
// =============================================================================

/// Per-user rolling statistics used to normalise structural signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserBaseline {
    /// Rolling mean message length in words.
    pub avg_message_length: f32,
    /// Rolling mean punctuation density (punct chars / total chars).
    pub avg_punctuation_density: f32,
    /// Number of messages observed so far.
    pub message_count: u32,
}

impl Default for UserBaseline {
    fn default() -> Self {
        Self {
            avg_message_length: 20.0,
            avg_punctuation_density: 0.05,
            message_count: 0,
        }
    }
}

/// Per-request context that may influence salience scores.
#[derive(Debug, Clone, Default)]
pub struct InteractionContext {
    /// Number of times the user has retried this request.
    pub retry_count: u32,
    /// Total messages in the current session.
    pub session_message_count: u32,
    /// Text of the immediately preceding message, if any.
    pub previous_text: Option<String>,
}

// =============================================================================
// SalienceAnalyzer
// =============================================================================

/// Computes emotional salience scores for incoming text.
pub struct SalienceAnalyzer {
    pub baseline: UserBaseline,
    #[cfg(feature = "onnx")]
    pub onnx_session: ort::session::Session,
}

impl SalienceAnalyzer {
    /// Create a new analyzer with a default baseline and no ONNX model.
    pub fn new() -> Self {
        Self {
            baseline: UserBaseline::default(),
            #[cfg(feature = "onnx")]
            onnx_session: todo!("call with_onnx() to initialise an ONNX session"),
        }
    }

    /// Create an analyzer backed by an ONNX emotion-classification model.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file on disk.
    ///
    /// TODO: initialise an `ort::Environment`, load the session from `model_path`,
    /// and store it in `onnx_session`.
    #[cfg(feature = "onnx")]
    pub fn with_onnx(_model_path: &str) -> Result<Self> {
        todo!("init ort env, load session from model_path")
    }

    /// Score the emotional salience of `text` given optional interaction context.
    ///
    /// Calls `score_structural` (always) and `score_onnx` (when the `onnx`
    /// feature is enabled), then blends them with `combine_scores`.
    /// Also updates the rolling baseline from this message.
    pub fn analyze(&mut self, text: &str, ctx: &InteractionContext) -> SalienceScore {
        let structural_score = todo_score_structural(self, text, ctx);
        let signals = build_signals(text, ctx, &self.baseline);

        #[cfg(feature = "onnx")]
        let (onnx_score, onnx_emotion) = self
            .score_onnx(text)
            .unwrap_or((structural_score, None));

        #[cfg(not(feature = "onnx"))]
        let (onnx_score, onnx_emotion): (f32, Option<OnnxEmotion>) = (structural_score, None);

        let value = self.combine_scores(structural_score, Some(onnx_score));

        // Derive valence from the most informative available signal.
        let valence = signals
            .explicit_emotion
            .as_ref()
            .map(|e| e.valence())
            .or_else(|| onnx_emotion.as_ref().map(|_| 0.0))
            .unwrap_or(0.0);

        let confidence = if cfg!(feature = "onnx") { 0.85 } else { 0.60 };

        self.update_baseline(text);

        SalienceScore {
            value,
            valence,
            confidence,
            signals: SalienceSignals {
                onnx_emotion,
                ..signals
            },
        }
    }

    /// Incrementally update the rolling mean baseline from `text`.
    pub fn update_baseline(&mut self, text: &str) {
        let n = self.baseline.message_count as f32;
        let new_n = n + 1.0;

        let word_count = text.split_whitespace().count() as f32;
        self.baseline.avg_message_length =
            (self.baseline.avg_message_length * n + word_count) / new_n;

        let punct_density = if text.is_empty() {
            0.0
        } else {
            text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32
                / text.chars().count() as f32
        };
        self.baseline.avg_punctuation_density =
            (self.baseline.avg_punctuation_density * n + punct_density) / new_n;

        self.baseline.message_count += 1;
    }

    /// Blend structural and ONNX scores into a single salience value.
    ///
    /// When no ONNX score is available the structural score is returned as-is
    /// (clamped to [0, 1]).
    pub fn combine_scores(&self, structural: f32, onnx: Option<f32>) -> f32 {
        match onnx {
            Some(o) => (structural * WEIGHT_STRUCTURAL + o * WEIGHT_ONNX).clamp(0.0, 1.0),
            None => structural.clamp(0.0, 1.0),
        }
    }

    /// Compute the structural salience score from heuristic signals.
    ///
    /// TODO: implement using the following signal weights:
    ///
    /// ```text
    /// punctuation_intensity × 0.15
    /// length_deviation      × 0.10
    /// correction_detected   → +0.25 flat
    /// urgency_detected      → +0.30 flat
    /// retry_count           → min(count × 0.10, 0.20)
    /// explicit_emotion      → EmotionTag::salience_weight()
    /// ```
    ///
    /// Normalise and clamp the result to [0, 1].
    pub fn score_structural(&self, _text: &str, _ctx: &InteractionContext) -> f32 {
        todo!("implement structural scoring using signal weights above")
    }

    /// Run the ONNX emotion classifier on `text` and return a salience weight
    /// plus the raw emotion output.
    ///
    /// TODO: tokenize `text`, run `onnx_session`, apply softmax, then map the
    /// winning label to a salience weight using:
    ///
    /// ```text
    /// anger/frustration → 0.85   fear     → 0.80
    /// joy               → 0.70   sadness  → 0.65
    /// surprise          → 0.55   neutral  → 0.15
    /// ```
    #[cfg(feature = "onnx")]
    pub fn score_onnx(&self, _text: &str) -> Result<(f32, Option<OnnxEmotion>)> {
        todo!("tokenize → run onnx_session → softmax → map label to salience_weight")
    }

    /// Persist the current baseline to the `salience_baselines` table.
    ///
    /// TODO: upsert a row with `(agent_id, user_id, JSON(self.baseline), NOW())`.
    pub async fn save_baseline(
        &self,
        _db: &DatabaseClient,
        _agent_id: &str,
        _user_id: Option<&str>,
    ) -> Result<()> {
        todo!("upsert baseline into salience_baselines")
    }

    /// Load a previously persisted baseline from the database, or return a
    /// default `SalienceAnalyzer` if no row exists.
    ///
    /// TODO: query `salience_baselines` for `(agent_id, user_id)`, deserialize
    /// `data` as `UserBaseline`, and construct `SalienceAnalyzer { baseline, .. }`.
    pub async fn load_baseline(
        _db: &DatabaseClient,
        _agent_id: &str,
        _user_id: Option<&str>,
    ) -> Result<Self> {
        todo!("query salience_baselines and deserialize UserBaseline")
    }
}

impl Default for SalienceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Private signal helpers (fully implemented — simple string matching)
// =============================================================================

/// Returns `true` if `text` contains any urgency word/phrase.
fn detect_urgency(text: &str) -> bool {
    let lower = text.to_lowercase();
    URGENCY_WORDS.iter().any(|w| lower.contains(w))
}

/// Returns `true` if `text` appears to correct the previous assistant turn.
fn detect_correction(text: &str, prev: Option<&str>) -> bool {
    let lower = text.to_lowercase();
    if CORRECTION_PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }
    // Also flag if the previous text was very short and the new one starts
    // with a negation — heuristic for "No, [correction]".
    if let Some(p) = prev {
        if p.len() < 30 && (lower.starts_with("no ") || lower.starts_with("no,")) {
            return true;
        }
    }
    false
}

/// Returns the most salient detected emotion in `text`, or `None` for neutral.
///
/// Priority: Frustration > Urgency > Excitement > Confusion > Satisfaction.
fn detect_explicit_emotion(text: &str) -> Option<EmotionTag> {
    let lower = text.to_lowercase();

    if FRUSTRATION_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(EmotionTag::Frustration);
    }
    if URGENCY_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(EmotionTag::Urgency);
    }
    if EXCITEMENT_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(EmotionTag::Excitement);
    }
    if CONFUSION_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(EmotionTag::Confusion);
    }
    if SATISFACTION_WORDS.iter().any(|w| lower.contains(w)) {
        return Some(EmotionTag::Satisfaction);
    }
    None
}

/// Compute punctuation intensity: proportion of punctuation chars relative to
/// message length, normalised against baseline density.
///
/// TODO: implement formula — e.g.
///   `(raw_density - baseline_density).max(0.0) / (1.0 - baseline_density).max(f32::EPSILON)`
fn compute_punctuation_intensity(_text: &str, _baseline: &UserBaseline) -> f32 {
    todo!(
        "compute (raw_punct_density - baseline_density).max(0) / (1 - baseline_density).max(ε)"
    )
}

/// Compute length deviation: how many standard deviations the message word-count
/// is above the user's rolling average.
///
/// TODO: implement formula — e.g.
///   `(word_count as f32 - avg_length).max(0.0) / avg_length.max(1.0)`
fn compute_length_deviation(_text: &str, _baseline: &UserBaseline) -> f32 {
    todo!("compute (word_count - avg_length).max(0) / avg_length.max(1)")
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Build a `SalienceSignals` struct from text + context (without onnx_emotion).
fn build_signals(
    text: &str,
    ctx: &InteractionContext,
    _baseline: &UserBaseline,
) -> SalienceSignals {
    let prev = ctx.previous_text.as_deref();
    SalienceSignals {
        punctuation_score: 0.0, // filled by score_structural once implemented
        length_deviation: 0.0,  // filled by score_structural once implemented
        correction_detected: detect_correction(text, prev),
        urgency_detected: detect_urgency(text),
        retry_count: ctx.retry_count,
        explicit_emotion: detect_explicit_emotion(text),
        onnx_emotion: None,
    }
}

/// Temporary shim that delegates to `score_structural` once implemented.
/// Until then, returns a best-effort score from detection helpers only.
fn todo_score_structural(analyzer: &SalienceAnalyzer, text: &str, ctx: &InteractionContext) -> f32 {
    let signals = build_signals(text, ctx, &analyzer.baseline);
    let mut score: f32 = 0.0;
    if signals.urgency_detected {
        score += 0.30;
    }
    if signals.correction_detected {
        score += 0.25;
    }
    score += (signals.retry_count as f32 * 0.10).min(0.20);
    if let Some(ref emotion) = signals.explicit_emotion {
        score = score.max(emotion.salience_weight());
    }
    score.clamp(0.0, 1.0)
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_urgency_detected() {
        assert!(detect_urgency("This is urgent and critical, please help"));
    }

    #[test]
    fn test_correction_detected() {
        assert!(detect_correction("No, that's wrong, I meant something else", None));
    }

    #[test]
    fn test_neutral_message_no_emotion() {
        assert!(detect_explicit_emotion("The weather is nice today").is_none());
    }

    #[test]
    fn test_excited_message_emotion() {
        let result = detect_explicit_emotion("This is amazing and perfect!");
        assert!(
            matches!(result, Some(EmotionTag::Excitement)),
            "expected Excitement, got {:?}",
            result
        );
    }

    #[test]
    fn test_update_baseline_running_average() {
        let mut analyzer = SalienceAnalyzer::new();
        // Baseline starts at avg_message_length = 20.0, count = 0
        analyzer.update_baseline("one two three four five"); // 5 words
        analyzer.update_baseline("a b c d e f g h i j k l m n o"); // 15 words
        // Incremental mean: (20*0 + 5)/1 = 5, then (5 + 15)/2 = 10
        assert!(
            (analyzer.baseline.avg_message_length - 10.0).abs() < 0.01,
            "expected avg ~10.0, got {}",
            analyzer.baseline.avg_message_length
        );
        assert_eq!(analyzer.baseline.message_count, 2);
    }

    #[test]
    fn test_combine_scores_weighted() {
        let analyzer = SalienceAnalyzer::new();
        // structural=0.4, onnx=0.8 → 0.4*0.45 + 0.8*0.55 = 0.18 + 0.44 = 0.62
        let result = analyzer.combine_scores(0.4, Some(0.8));
        assert!(
            (result - 0.62).abs() < 0.001,
            "expected ~0.62, got {}",
            result
        );
    }
}
