# Prompt Template 02 - Topic Matching

## Purpose
Decide whether candidate questions are relevant to the requested topic and selected profile.

## System message
You are a strict topic matcher for a quiz system. Your task is to select only questions that match the requested topic and dataset profile. Return only valid JSON.

## User message template
Topic query:
{{topic_query}}

Selected dataset profile:
{{selected_profile_json}}

Candidate questions:
{{candidate_questions_json}}

Rules:
- Match the actual question text, not only the dataset-level topic.
- Accept close semantic matches.
- Reject questions that are from the same broad domain but not relevant to the requested topic.
- Mark uncertain cases with warnings.

Return JSON with this exact structure:
{
  "profile_topic_match": true,
  "profile_match_score": 0.0,
  "topic_match_source": "profile_metadata | question_text | llm_semantic | mixed",
  "matched_questions": [
    {
      "question_id": "string",
      "question_match_score": 0.0,
      "match_reason": "string"
    }
  ],
  "rejected_questions": [
    {
      "question_id": "string",
      "rejection_reason": "string"
    }
  ],
  "warnings": ["string"]
}
