//! Prompt templates for LLM interactions.
//!
//! These templates can be customized to adjust how the LLM processes
//! memories and summarization tasks.

/// Template for the summarization task instructions.
/// 
/// Placeholders:
/// - `{event_count}` - Number of events to summarize
/// - `{events_text}` - Formatted list of events
/// - `{agent_id}` - The agent identifier
/// - `{event_ids_json}` - JSON array of event IDs to mark as summarized
pub const SUMMARIZE_INSTRUCTIONS: &str = r#"## Summarization Task

I found {event_count} unsummarized event(s) that need to be consolidated into long-term memory.

### Events to Summarize:
{events_text}

### Instructions:
1. **Analyze** the events above and identify key information worth remembering:
   - User preferences and decisions
   - Project context and conventions  
   - Important facts or requirements
   - Patterns in user behavior

2. **Call `memento.store`** for each distinct piece of information you want to preserve:
   - Use descriptive, searchable text
   - Set appropriate `event_type`: "preference", "context", "decision", etc.
   - Add relevant tags in metadata

3. **Call `memento.mark_summarized`** with the event IDs to mark them as processed:
   ```json
   {
     "agent_id": "{agent_id}",
     "event_ids": {event_ids_json}
   }
   ```

This prevents these events from being re-processed in future summarization calls."#;

/// Format the summarization instructions with the given parameters.
pub fn format_summarize_instructions(
    event_count: usize,
    events_text: &str,
    agent_id: &str,
    event_ids: &[String],
) -> String {
    let event_ids_json = serde_json::to_string(event_ids).unwrap_or_else(|_| "[]".to_string());
    
    SUMMARIZE_INSTRUCTIONS
        .replace("{event_count}", &event_count.to_string())
        .replace("{events_text}", events_text)
        .replace("{agent_id}", agent_id)
        .replace("{event_ids_json}", &event_ids_json)
}
