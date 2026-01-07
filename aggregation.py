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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from example import LLMProvider, LLMResponse


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

    Returns:
        (prompt_string, models_in_context_ordered)
    """
    # Filter valid responses and randomize order
    valid_responses = [r for r in responses if r.reasoning]
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
Provide your reasoning, then state your final answer as: ANSWER: [A/B/C/D]"""

    return prompt, models_in_context


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the final answer from a response.

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
) -> AggregationResult:
    """Aggregate multiple model responses using blind peer review.

    Args:
        question: The question text
        native_responses: List of dicts with 'model', 'answer', 'reasoning' keys
        llm_provider: LLM provider for making API calls
        aggregator_model: Model to use for aggregation (if None, uses first model)
        correct_answer: Optional ground truth for scoring

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
    prompt, models_in_context = build_blind_aggregation_prompt(question, responses)

    # Use first model as aggregator if not specified
    if aggregator_model is None:
        aggregator_model = responses[0].model if responses else "gpt-4"

    # Call the aggregator model
    response = llm_provider.call(aggregator_model, prompt)

    # Extract the answer
    answer = extract_answer(response.text)

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
