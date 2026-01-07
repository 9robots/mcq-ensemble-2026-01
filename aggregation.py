"""
Aggregation Algorithm - Blind Peer Review

Reference implementation from 9robots benchmark pipeline.
Core algorithm for multi-model answer synthesis.

Design principles:
- Anonymous responses (no model names shown)
- Randomized order (prevent position bias)
- Full reasoning included (no truncation)
- Evaluate reasoning quality, not popularity
"""

import random
import re
import warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from example import LLMProvider, LLMResponse

# Default seed for reproducibility (set to None for random behavior)
DEFAULT_SEED = 42


@dataclass
class NativeResponse:
    """A single model's native response."""
    model: str
    answer: str
    reasoning: str


@dataclass
class AggregationResult:
    """Result of the aggregation process."""
    answer: str
    reasoning: str
    is_correct: Optional[bool] = None
    models_in_context: List[str] = None


def build_blind_aggregation_prompt(
    question: str,
    responses: List[NativeResponse],
    seed: Optional[int] = DEFAULT_SEED,
) -> Tuple[str, List[str]]:
    """Build a blind aggregation prompt WITHOUT authority bias.

    Key design decisions:
    - NO model names shown (anonymous)
    - NO accuracy scores (prevents deference to "experts")
    - Randomized order (prevents first-response anchoring)
    - Full reasoning included (no truncation)

    Args:
        question: The question text
        responses: List of native responses from models
        seed: Random seed for reproducibility (None for random)

    Returns:
        (prompt_string, models_in_context_ordered)
    """
    # Warn about empty reasoning (not silently filtered)
    empty_reasoning = [r for r in responses if not r.reasoning]
    if empty_reasoning:
        warnings.warn(
            f"{len(empty_reasoning)} response(s) have empty reasoning and will be "
            f"included with placeholder text. Models: {[r.model for r in empty_reasoning]}"
        )

    # Use all responses (don't filter) but mark empty ones
    valid_responses = list(responses)

    # Reproducible shuffle with seed
    if seed is not None:
        random.seed(seed)
    random.shuffle(valid_responses)

    # Build anonymous context
    context_parts = []
    models_in_context = []

    for i, r in enumerate(valid_responses, 1):
        models_in_context.append(r.model)
        reasoning = r.reasoning if r.reasoning else "(no reasoning provided)"
        context_parts.append(
            f"[Response {i}]\n"
            f"Answer: {r.answer}\n"
            f"Reasoning: {reasoning}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""PEER REVIEW TASK

You previously answered this question independently. Now review how other AI systems approached it.

IMPORTANT:
- Evaluate reasoning QUALITY, not popularity or reputation
- A minority view with rigorous logic may be correct
- You may maintain your answer if your reasoning is sound
- You may revise if another argument is more compelling

---

QUESTION:
{question}

---

PEER RESPONSES (anonymous, randomized order):

{context_text}

---

YOUR TASK:
1. Review each response's reasoning
2. Identify the strongest and weakest arguments
3. Decide whether to maintain or revise your answer

If MAINTAINING: Explain why your original reasoning holds despite alternatives.
If REVISING: Explain which argument convinced you and why.

YOUR RESPONSE:
Provide your reasoning and decision."""

    return prompt, models_in_context


def build_extract_answer_prompt(question: str, response: str) -> str:
    """Build prompt to extract concrete letter answer from verbose response.

    This is the follow-up call in the double-call pattern.
    """
    return f"""{question}

Your response was:
{response}

Based on your response above, what is your final answer? Reply with only the letter: A, B, C, or D."""


def extract_answer_with_followup(
    question: str,
    response_text: str,
    llm_provider: 'LLMProvider',
    model: str,
) -> Optional[str]:
    """Extract answer using follow-up call (double-call pattern).

    Instead of parsing the verbose response, we ask the model directly
    what letter answer it intended. This avoids formatting tax issues.
    """
    prompt = build_extract_answer_prompt(question, response_text)
    response = llm_provider.call(model, prompt)

    # Simple extraction from follow-up (should be just the letter)
    text = response.text.strip().upper()

    if text in ('A', 'B', 'C', 'D'):
        return text

    # Letter with punctuation
    if len(text) >= 1 and text[0] in ('A', 'B', 'C', 'D'):
        return text[0]

    # Pattern matching for slightly verbose responses
    patterns = [
        r'(?:the\s+)?answer\s*(?:is)?[:\s]+([ABCD])\b',
        r'\b([ABCD])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the final answer from a response (legacy regex-only).

    DEPRECATED: Use extract_answer_with_followup for double-call pattern.

    Looks for patterns like:
    - ANSWER: A
    - Answer: B
    - FINAL ANSWER: C
    - The answer is D
    """
    # Primary pattern: ANSWER: X or FINAL ANSWER: X
    patterns = [
        r'(?:FINAL\s+)?ANSWER:\s*\[?([A-D])\]?',
        r'(?:the\s+)?answer\s+is\s*:?\s*\[?([A-D])\]?',
        r'\b([A-D])\s*(?:is\s+(?:the\s+)?(?:correct|right|best)\s+answer)',
    ]

    text = response_text.upper()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: last standalone letter A-D in the response
    matches = re.findall(r'\b([A-D])\b', text[-200:])
    if matches:
        return matches[-1]

    return None


def aggregate_responses(
    question: str,
    native_responses: List[Dict],
    llm_provider: LLMProvider,
    aggregator_model: str = None,
    correct_answer: str = None,
    seed: Optional[int] = DEFAULT_SEED,
) -> AggregationResult:
    """Aggregate multiple model responses using blind peer review.

    Args:
        question: The question text
        native_responses: List of dicts with 'model', 'answer', 'reasoning' keys
        llm_provider: LLM provider for making API calls
        aggregator_model: Model to use for aggregation (if None, uses first model)
        correct_answer: Optional ground truth for scoring
        seed: Random seed for reproducibility (None for random)

    Returns:
        AggregationResult with synthesized answer and reasoning
    """
    # Convert to NativeResponse objects
    responses = [
        NativeResponse(
            model=r.get('model', 'unknown'),
            answer=r.get('answer', ''),
            reasoning=r.get('reasoning', ''),
        )
        for r in native_responses
    ]

    # Build the aggregation prompt
    prompt, models_in_context = build_blind_aggregation_prompt(question, responses, seed=seed)

    # Use first model as aggregator if not specified
    if aggregator_model is None:
        aggregator_model = responses[0].model if responses else "gpt-4"

    # Call the aggregator model (reasoning call)
    response = llm_provider.call(aggregator_model, prompt)

    # Extract the answer using follow-up call (double-call pattern)
    answer = extract_answer_with_followup(
        question=question,
        response_text=response.text,
        llm_provider=llm_provider,
        model=aggregator_model,
    )

    # Score if ground truth provided
    is_correct = None
    if correct_answer and answer:
        is_correct = answer.upper() == correct_answer.upper()

    return AggregationResult(
        answer=answer,
        reasoning=response.text,
        is_correct=is_correct,
        models_in_context=models_in_context,
    )


# Example usage
if __name__ == "__main__":
    from example import OpenAIProvider

    # Initialize provider
    llm = OpenAIProvider(api_key="your-api-key")

    # Example aggregation
    result = aggregate_responses(
        question="What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid",
        native_responses=[
            {"model": "gpt-4", "answer": "B", "reasoning": "Paris is the capital of France..."},
            {"model": "claude-3", "answer": "B", "reasoning": "The capital of France is Paris..."},
            {"model": "gemini", "answer": "A", "reasoning": "London is a major European city..."},
        ],
        llm_provider=llm,
        correct_answer="B",
    )

    print(f"Aggregated answer: {result.answer}")
    print(f"Correct: {result.is_correct}")
