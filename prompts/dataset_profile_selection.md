# Prompt Template 01 - Dataset Profile Selection

## Purpose
Select the most suitable dataset profile for a requested quiz.

## System message
You are a strict dataset-profile selector for a quiz system. Your task is to select a dataset profile only when it is compatible with the user's request. You must not invent datasets or capabilities. Return only valid JSON.

## User message template
User request:
{{user_request}}

Available dataset profiles:
{{dataset_profiles_json}}

Selection constraints:
- Respect explicit dataset_key if provided.
- Respect explicit domain, topic, question_type, answer_type, and context requirement if provided.
- Prefer profiles with matching topic and evaluator compatibility.
- If no suitable profile exists, return selected_profile as null and explain why.
- Do not silently select an incompatible profile.

Return JSON with this exact structure:
{
  "selected_profile": "string or null",
  "confidence": 0.0,
  "completed_user_request": {
    "topic_query": "string or null",
    "dataset_key": "string or null",
    "domain": "string or null",
    "question_type": "string or null",
    "answer_type": "string or null",
    "requires_context": true
  },
  "matched_fields": ["string"],
  "missing_fields": ["string"],
  "compatibility_checks": {
    "topic_supported": true,
    "question_type_supported": true,
    "answer_type_supported": true,
    "context_requirement_supported": true,
    "reference_answer_available": true,
    "recommended_evaluator_available": true
  },
  "selection_reason": "string",
  "warnings": ["string"]
}
