# MCQ Ensemble Reference Implementation

Reference implementation for the paper: **"Consistently Not Stupid: Multi-Model Aggregation for Error Reduction in LLMs"** (January 2025)

This folder contains standalone implementations of the core algorithms, extracted from the production benchmark pipeline.

## Overview

Two algorithms for multi-model answer synthesis:

1. **Aggregation** (`aggregation.py`): Blind peer review where models evaluate each other's reasoning without knowing model identities.

2. **Debate** (`debate.py`): Multi-round structured debate for disagreement cases, using steelman/critique/decision format.

## Quick Start

```python
from aggregation import aggregate_responses
from debate import run_debate
from example import YourLLMProvider

# Initialize your LLM provider
llm = YourLLMProvider(api_key="...")

# Aggregation: synthesize answer from multiple model responses
result = aggregate_responses(
    question="What is the capital of France?",
    native_responses=[
        {"answer": "A", "reasoning": "Paris is the capital..."},
        {"answer": "A", "reasoning": "The capital of France is Paris..."},
        {"answer": "B", "reasoning": "Lyon is the largest city..."},
    ],
    llm_provider=llm
)
print(result["answer"])  # Aggregated answer

# Debate: resolve disagreement through structured deliberation
result = run_debate(
    question="Complex science question...",
    native_responses=[...],  # Mixed answers
    models=["model_a", "model_b", "model_c"],
    llm_provider=llm,
    max_rounds=5
)
print(result["final_answer"])
print(result["consensus_reached"])
```

## Files

| File | Description |
|------|-------------|
| `aggregation.py` | Peer review aggregation algorithm |
| `debate.py` | Multi-round debate with steelman/critique structure |
| `example.py` | Abstract `LLMProvider` interface for custom backends |
| `prompts/` | All prompt templates |

## Algorithm Details

### Aggregation (Blind Peer Review)

1. Collect native responses from N models
2. Shuffle response order (prevent position bias)
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

## Adapting to Your Infrastructure

Implement the `LLMProvider` interface in `example.py`:

```python
class YourLLMProvider(LLMProvider):
    def call(self, model: str, prompt: str) -> str:
        # Your API call here
        return response_text
```

## Relationship to Production Code

This reference implementation is extracted from the production pipeline in `src/pipeline/`. The production code includes:
- Database persistence
- Parallel execution with ThreadPoolExecutor
- Retry logic and error handling
- Cost tracking and metrics

This reference strips all infrastructure concerns to show the core algorithms clearly.

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
