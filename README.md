RAG_BASE/
│
├── data/
│   ├── raw/
│   │   Raw local quiz datasets in JSONL format.
│   │   These files may have different original formats.
│   │
│   ├── processed/
│   │   Unified datasets after conversion.
│   │   All samples should follow the same schema.
│   │
│   └── results/
│       Evaluation outputs produced by the framework.
│
├── prompts/
│   Prompt templates for different evaluation approaches.
│   For example:
│   - LLM-as-a-Judge
│   - RAG-based judge
│   - rubric-based judge
│   - verifier
│
├── src/
│   ├── schemas.py
│   │   Defines the unified quiz evaluation format.
│   │   All datasets are converted into this schema.
│   │
│   ├── config.py
│   │   Stores configuration such as model names and paths.
│   │
│   ├── main.py
│   │   Entry point of the framework.
│   │   Loads data, converts it, routes samples, evaluates them, and saves results.
│   │
│   ├── data/
│   │   ├── adapters.py
│   │   │   Converts local JSONL datasets into the unified schema.
│   │   │
│   │   └── hf_adapters.py
│   │       Loads Hugging Face datasets and converts them into the unified schema.
│   │
│   ├── core/
│   │   ├── router.py
│   │   │   Chooses the evaluation approach based on question type and context.
│   │   │
│   │   ├── evaluator.py
│   │   │   Runs the selected evaluation method.
│   │   │
│   │   ├── scorer.py
│   │   │   Computes or aggregates final scores.
│   │   │
│   │   ├── retrieval.py
│   │   │   Retrieves evidence documents or context chunks.
│   │   │
│   │   ├── verifier.py
│   │   │   Optional second-pass validation.
│   │   │
│   │   └── aggregator.py
│   │       Combines multiple scores or judge outputs.
│   │
│   ├── llm/
│   │   ├── client.py
│   │   │   Handles OpenAI API calls.
│   │   │
│   │   └── prompt_loader.py
│   │       Loads prompt templates from the prompts folder.
│   │
│   └── utils/
│       ├── jsonl.py
│       │   Helper functions for reading and writing JSONL files.
│       │
│       └── text.py
│           Text processing helper functions.


# AI / RAG-Based Quiz Evaluation Framework

## Goal

This project investigates AI-based and RAG-based approaches for quiz evaluation.

The goal is to build a modular framework that can evaluate different types of quiz answers using different evaluation strategies, such as:

- rule-based evaluation
- semantic evaluation
- LLM-as-a-Judge
- RAG-based evaluation
- rubric-based evaluation
- multi-stage evaluation

## Current Focus

The current focus is not only to design individual prompts, but to build a higher-level framework for quiz evaluation.

The framework should answer:

1. What type of question is this?
2. Is context or retrieved evidence needed?
3. Which evaluation approach should be used?
4. Which scoring dimensions are relevant?
5. How should the final score and explanation be produced?

## Framework Pipeline

Raw dataset  
→ Dataset adapter  
→ Unified quiz schema  
→ Task analyzer / router  
→ Evaluation method  
→ Scoring and explanation  
→ Result JSON

## Unified Schema

All datasets are converted into one unified format:

- id
- dataset
- question_type
- question
- student_answer
- reference_answer
- context
- options
- requires_context
- rubric
- metadata

## Dataset Adapters

There are two types of adapters:

### Local dataset adapters

Defined in:

`src/data/adapters.py`

Used for local JSONL files in:

`data/raw/`

### Hugging Face dataset adapters

Defined in:

`src/data/hf_adapters.py`

Used for datasets such as:

- HotpotQA
- BoolQ
- Natural Questions
- RAGBench

## Evaluation Approaches

The framework will support different approaches:

### Rule-based / Exact Match

Used for MCQ and true/false questions.

### LLM-as-a-Judge

Used for short answers without strong context requirements.

### RAG-Based Evaluation

Used when provided or retrieved context is required.

### Rubric-Centric RAG

Combines context, reference answer, and structured rubric.

### Multi-Stage Evaluation

Possible production-style pattern:

retrieval  
→ filtering  
→ LLM scoring  
→ optional verification

## Current Implementation Plan

This week, the goal is to implement a minimal prototype:

1. Convert local JSONL samples into the unified schema.
2. Route each sample to an evaluation method.
3. Run rule-based evaluation for MCQ / true-false.
4. Run RAG/rubric-based LLM evaluation for context-based short answers.
5. Save structured evaluation results.
6. Later, connect Hugging Face datasets through `hf_adapters.py`.

## This Week's Main Task

Build the framework structure first.

The priority is:

- unified schema
- dataset adapter layer
- router
- basic evaluator
- OpenAI API connection
- small demo on local samples

Hugging Face datasets will be added after the local pipeline works.