# src/evaluation/prompt_sensitivity_prompts.py

from __future__ import annotations


BASIC_LENIENT_PROMPT = """
You are a semantic answer evaluator.

Your task is to evaluate whether the student's answer expresses the same core meaning as the reference answer.

Evaluation style:
- Be reasonably lenient.
- Accept paraphrases.
- Accept concise answers if they contain the core meaning.
- Accept minor wording differences.
- Do not require the student answer to use the exact same words as the reference answer.
- Penalize answers that are clearly wrong, irrelevant, or contradictory.

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Return ONLY valid JSON in this exact format:
{{
  "score": <number between 0.0 and 1.0>,
  "passed": <true if score >= 0.5 else false>,
  "feedback": "<brief explanation of the score>"
}}
""".strip()


STRICT_PROMPT = """
You are a strict semantic answer evaluator.

Your task is to evaluate the student's answer strictly according to the reference answer.

Evaluation style:
- Award 1.0 only if the answer is fully correct and includes the key required information.
- Award 0.0 if the answer is incorrect, incomplete, ambiguous, irrelevant, contradictory, or only partially correct.
- Do not give partial scores.
- Do not infer missing information from the student's answer.
- Penalize unsupported extra claims if they change, weaken, or confuse the answer.

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Return ONLY valid JSON in this exact format:
{{
  "score": 0.0 or 1.0,
  "passed": true or false,
  "feedback": "<brief explanation of the score>",
  "reason_for_deduction": "<null if no deduction, otherwise brief reason>"
}}
""".strip()


RUBRIC_BASED_PROMPT = """
You are a rubric-based semantic answer evaluator.

Your task is to evaluate the student's answer using four explicit dimensions.

Dimensions:

1. semantic_correctness:
Does the answer express the core meaning of the reference answer?
Score from 0.0 to 1.0.

2. completeness:
Does the answer include the key required information?
Score from 0.0 to 1.0.

3. relevance:
Does the answer directly answer the question?
Score from 0.0 to 1.0.

4. no_contradiction:
Does the answer avoid contradicting the reference answer?
Score from 0.0 to 1.0.

Final score:
Compute the final score as the average of the four dimensions.

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Return ONLY valid JSON in this exact format:
{{
  "semantic_correctness": <number between 0.0 and 1.0>,
  "completeness": <number between 0.0 and 1.0>,
  "relevance": <number between 0.0 and 1.0>,
  "no_contradiction": <number between 0.0 and 1.0>,
  "score": <average of the four dimensions>,
  "passed": <true if score >= 0.5 else false>,
  "feedback": "<brief explanation of the score>"
}}
""".strip()


PROMPT_VARIANTS = {
    "basic_lenient": {
        "prompt": BASIC_LENIENT_PROMPT,
        "style": "lenient",
        "score_type": "continuous",
        "purpose": "Tests a more forgiving semantic evaluator prompt.",
    },
    "strict": {
        "prompt": STRICT_PROMPT,
        "style": "strict",
        "score_type": "binary",
        "purpose": "Tests a strict evaluator prompt that does not allow partial credit.",
    },
    "rubric_based": {
        "prompt": RUBRIC_BASED_PROMPT,
        "style": "rubric_based",
        "score_type": "multi_dimensional",
        "purpose": "Tests a structured evaluator prompt with explicit scoring dimensions.",
    },
}


TEST_CASES = [
    {
        "id": "TC1",
        "type": "exact_correct",
        "question": "What is the capital of France?",
        "reference_answer": "Paris.",
        "student_answer": "Paris.",
        "expected_behavior": "All prompts should give a high score.",
    },
    {
        "id": "TC2",
        "type": "correct_with_extra_information",
        "question": "What is the capital of France?",
        "reference_answer": "Paris.",
        "student_answer": "Paris is the capital and largest city of France.",
        "expected_behavior": "The answer is correct but includes extra information.",
    },
    {
        "id": "TC3",
        "type": "clearly_wrong",
        "question": "What is the capital of France?",
        "reference_answer": "Paris.",
        "student_answer": "Lyon.",
        "expected_behavior": "All prompts should give a low score.",
    },
    {
        "id": "TC4",
        "type": "verbose_correct",
        "question": "What causes seasons on Earth?",
        "reference_answer": "The tilt of Earth's axis causes different parts of Earth to receive different amounts of sunlight during the year.",
        "student_answer": "The Earth's axis is tilted relative to its orbit around the Sun, so different parts of Earth receive more direct sunlight at different times of the year. This causes the seasons.",
        "expected_behavior": "All prompts should give a high score, although exact scores may differ.",
    },
    {
        "id": "TC5",
        "type": "misconception",
        "question": "What causes seasons on Earth?",
        "reference_answer": "The tilt of Earth's axis causes different parts of Earth to receive different amounts of sunlight during the year.",
        "student_answer": "Seasons happen because the Earth is closer to the Sun in summer and farther away from the Sun in winter.",
        "expected_behavior": "All prompts should give a low score because the answer contains a misconception.",
    },
    {
        "id": "TC6",
        "type": "partial_answer",
        "question": "What causes seasons on Earth?",
        "reference_answer": "The tilt of Earth's axis causes different parts of Earth to receive different amounts of sunlight during the year.",
        "student_answer": "Different parts of Earth receive different amounts of sunlight during the year.",
        "expected_behavior": "This is a borderline answer because it omits the key cause: Earth's axial tilt.",
    },
    {
        "id": "TC7",
        "type": "irrelevant_answer",
        "question": "What causes seasons on Earth?",
        "reference_answer": "The tilt of Earth's axis causes different parts of Earth to receive different amounts of sunlight during the year.",
        "student_answer": "The Earth rotates once every 24 hours.",
        "expected_behavior": "All prompts should give a low score due to irrelevance.",
    },
]