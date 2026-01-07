# MCQ Ensemble Reference Implementation

Reference implementation for the paper: **"Consistently Not Stupid: Multi-Model Aggregation for Error Reduction in LLMs"** (January 2025)

This folder contains standalone implementations of the core algorithms, extracted from the production benchmark pipeline.

## Installation

```bash
pip install -r requirements.txt
```

## Overview

Three core components for multi-model answer synthesis:

1. **Collection** (`collection.py`): Double-call protocol that separates verdict (answer) from rationale (explanation) to avoid the "formatting tax" observed with GPT-5.2.

2. **Aggregation** (`aggregation.py`): Blind peer review where models evaluate each other's reasoning without knowing model identities.

3. **Debate** (`debate.py`): Multi-round structured debate for disagreement cases, using steelman/critique/decision format.

## Quick Start

```python
from collection import collect_with_double_call
from aggregation import aggregate_responses
from debate import run_debate
from example import OpenAIProvider  # or AnthropicProvider

# Initialize your LLM provider
# CRITICAL: Use reasoning_effort="xhigh" for GPT-5.2
llm = OpenAIProvider(api_key="your-api-key", reasoning_effort="xhigh")

# Collection: get verdict and rationale using double-call protocol
result = collect_with_double_call(
    question="What is 2 + 2?\nA) 3\nB) 4\nC) 5\nD) 6",
    llm_provider=llm,
    model="gpt-5.2",
    correct_answer="B",
)
print(result.verdict)    # "B"
print(result.rationale)  # Explanation for aggregation context
print(result.is_correct) # True

# Aggregation: synthesize answer from multiple model responses
result = aggregate_responses(
    question="What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid",
    native_responses=[
        {"model": "gpt-4", "answer": "B", "reasoning": "Paris is the capital..."},
        {"model": "claude-3", "answer": "B", "reasoning": "The capital of France is Paris..."},
        {"model": "gemini", "answer": "A", "reasoning": "London is a major European city..."},
    ],
    llm_provider=llm,
    aggregator_model="gpt-4",  # Model to use for aggregation
    seed=42,  # For reproducibility (None for random)
)
print(result.answer)  # Aggregated answer
print(result.is_correct)  # If correct_answer was provided

# Debate: resolve disagreement through structured deliberation
result = run_debate(
    question="Complex science question...\nA) ... B) ... C) ... D) ...",
    native_responses=[
        {"model": "gpt-4", "answer": "A", "reasoning": "Because..."},
        {"model": "claude-3", "answer": "B", "reasoning": "Because..."},
        {"model": "gemini", "answer": "A", "reasoning": "Because..."},
    ],
    models=["gpt-4", "claude-3", "gemini"],
    llm_provider=llm,
    max_rounds=5,
)
print(result.final_answer)
print(result.consensus_reached)
print(result.rounds_taken)
```

## Files

| File | Description |
|------|-------------|
| `collection.py` | Double-call protocol for collecting verdicts and rationales |
| `aggregation.py` | Peer review aggregation algorithm |
| `debate.py` | Multi-round debate with steelman/critique structure |
| `example.py` | Abstract `LLMProvider` interface with OpenAI/Anthropic/Gemini examples |
| `model_configs.py` | Exact model configurations for benchmark reproducibility |
| `prompts/` | All prompt templates |
| `requirements.txt` | Python dependencies |

## Algorithm Details

### Collection (Double-Call Protocol)

Separates verdict (answer) from rationale (explanation) to avoid the "formatting tax":

**The Problem**: GPT-5.2 with `reasoning_effort="xhigh"` achieves 92.4% on GPQA Diamond with OpenAI's simple prompt. However, compound prompts that ask the model to both reason AND format (e.g., "Think step-by-step... then state ANSWER: [letter]") degrade accuracy by ~15pp. Other models (Claude, Gemini) are resilient to this effect.

**The Solution**:
1. **Call A (Verdict)**: Simple prompt requesting only the answer letter
   - "Answer with only the letter: A, B, C, or D."
   - Model uses internal reasoning (xhigh, extended thinking, etc.)
2. **Call B (Rationale)**: Separate prompt requesting explanation
   - "You answered [X]. Explain your reasoning for this answer."
   - Provides context for aggregation stage

**Trade-off**: 2x API cost during collection, but eliminates formatting tax and parsing complexity.

### Aggregation (Blind Peer Review)

1. Collect native responses from N models
2. Shuffle response order with reproducible seed (prevent position bias)
3. Present anonymized responses to aggregator model
4. Model synthesizes answer based on reasoning quality, not popularity

Key design: Models see reasoning but not model identities, preventing authority bias.

### Debate (Structured Deliberation)

Triggered only for **disagreement cases** (non-unanimous native answers).

Each round requires three steps:
1. **STEELMAN**: Articulate the strongest version of opposing arguments
2. **CRITIQUE**: Identify specific invalid inferential steps with evidence
3. **DECISION**: State final answer with justification

Termination:
- **Consensus**: All models agree
- **Stuck**: No position changes for 2 rounds
- **Max rounds**: Safety limit (default: 5)

If stuck, models vote independently after reviewing full debate history.

## Model Configuration

**CRITICAL**: API defaults often differ from vendor benchmark settings! See `model_configs.py` for exact configurations.

```python
# GPT-5.2: default reasoning_effort="none" gives ~75%
# Use "xhigh" for vendor-reported 92.4%
llm = OpenAIProvider(api_key="...", reasoning_effort="xhigh")

# Gemini 3 Pro: explicitly set HIGH thinking for 91.9%
llm = GeminiProvider(project_id="...", thinking_level="HIGH")

# Claude: extended thinking automatic, no parameter needed
llm = AnthropicProvider(api_key="...")
```

## Adapting to Your Infrastructure

Implement the `LLMProvider` interface in `example.py`:

```python
from example import LLMProvider, LLMResponse

class YourLLMProvider(LLMProvider):
    def call(self, model: str, prompt: str) -> LLMResponse:
        # Your API call here
        response_text = your_api.call(model, prompt)
        return LLMResponse(text=response_text, model=model)
```

## Differences from Production Code

This reference implementation is simplified for clarity. Key differences:

| Feature | Reference | Production |
|---------|-----------|------------|
| Collection | Double-call protocol | Same |
| Answer Extraction | Simple letter extraction | Same (double-call makes this trivial) |
| Execution | Sequential | Parallel (ThreadPoolExecutor) |
| Persistence | None | PostgreSQL |
| Error Handling | Basic | Retry with backoff |
| Cost Tracking | None | Full token/cost accounting |

**Note**: The double-call protocol eliminates the need for complex answer extraction. Since Call A only asks for a letter, extraction is trivial.

## Reproducibility

The `seed` parameter ensures deterministic response ordering:

```python
# Same seed = same order = reproducible results
result1 = aggregate_responses(..., seed=42)
result2 = aggregate_responses(..., seed=42)
# result1 and result2 will have identical prompt construction

# Different seed or None = different order
result3 = aggregate_responses(..., seed=None)  # Random each time
```

## Citation

```bibtex
@article{razumny2025ensemble,
  title={Consistently Not Stupid: Multi-Model Aggregation for Error Reduction in LLMs},
  author={Razumny, Igor},
  year={2025},
  url={https://benchmark.9robots.ai}
}
```

## License

MIT License
