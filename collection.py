"""
Double-Call Collection Protocol

Reference implementation from 9robots benchmark pipeline.
Separates verdict (answer) from rationale (explanation) to avoid "formatting tax".

The Problem:
    GPT-5.2 with reasoning_effort="xhigh" achieves 92.4% on GPQA Diamond
    when using OpenAI's simple prompt format. However, compound prompts that
    ask the model to both reason AND format (e.g., "Think step-by-step...
    then state ANSWER: [letter]") degrade accuracy by ~15pp.

    Other models (Claude, Gemini) are resilient to this effect.

The Solution - Double-Call Protocol:
    Call A (Verdict): Simple prompt requesting only the answer letter
        - Uses model's internal reasoning (xhigh, extended thinking, etc.)
        - Avoids formatting tax from compound prompts

    Call B (Rationale): Separate prompt requesting explanation
        - Provides context for aggregation stage
        - Model explains why it chose the answer

This approach:
    - Eliminates formatting tax
    - Removes parsing complexity (just extract single letter)
    - Measures "peak reasoning" rather than "instruction compliance"
    - Trade-off: 2x API cost during collection phase
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass

from example import LLMProvider, LLMResponse


@dataclass
class CollectionResult:
    """Result from double-call collection."""
    verdict: str  # A, B, C, or D
    rationale: str  # Explanation for aggregation context
    is_correct: Optional[bool] = None
    # Combined metrics from both calls
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def strip_cot_instructions(prompt_text: str) -> str:
    """Strip Chain-of-Thought instructions from prompt.

    Many benchmark prompts include CoT suffixes like:
        "Think through this step-by-step:
        1. Analyze what the question is asking
        2. Consider each option...
        ...
        ANSWER: [letter]"

    We strip these to create clean prompts for double-call.
    """
    pattern = re.compile(
        r'\n\nThink through this step-by-step:.*?ANSWER:\s*\[letter\]',
        re.DOTALL | re.IGNORECASE
    )
    return pattern.sub('', prompt_text).strip()


def make_verdict_prompt(question: str) -> str:
    """Create Call A (Verdict) prompt - simple answer request.

    Uses model's internal reasoning (xhigh, extended thinking, etc.)
    but only asks for the letter output. Avoids formatting tax.
    """
    base = strip_cot_instructions(question)
    return f"{base}\n\nAnswer with only the letter: A, B, C, or D."


def make_rationale_prompt(question: str, verdict: str) -> str:
    """Create Call B (Rationale) prompt - explanation request.

    Asks model to explain why it chose the answer.
    Provides context for peer review in aggregation stage.
    """
    base = strip_cot_instructions(question)
    return f"{base}\n\nYou answered {verdict}. Explain your reasoning for this answer."


def extract_verdict(response_text: str) -> Optional[str]:
    """Extract verdict (A/B/C/D) from simple response.

    The verdict prompt asks for just the letter, so extraction is simple.
    We expect responses like: "D", "D.", "The answer is D", etc.
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Direct single letter (case insensitive)
    if text.upper() in ('A', 'B', 'C', 'D'):
        return text.upper()

    # Letter with period or colon: "D." or "D:"
    if len(text) >= 1 and text[0].upper() in ('A', 'B', 'C', 'D'):
        if len(text) == 1 or text[1] in '.):,':
            return text[0].upper()

    # Look for explicit answer patterns (order matters - more specific first)
    patterns = [
        r'(?:the\s+)?answer\s*(?:is)?[:\s]+([ABCD])\b',  # "Answer: D" or "The answer is D"
        r'\b([ABCD])\s*(?:is\s+(?:the\s+)?(?:correct|right|answer))',  # "D is correct"
        r'(?:option|choice)\s+([ABCD])\b',  # "Option D"
        r'^([ABCD])\s*$',  # Just the letter at start/end
        r'^([ABCD])[.):,\s]',  # Letter with punctuation at start
        r'[.):,\s]([ABCD])$',  # Letter at end after punctuation
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find isolated letter (not part of a word)
    match = re.search(r'(?<![A-Za-z])([ABCD])(?![A-Za-z])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def collect_with_double_call(
    question: str,
    llm_provider: LLMProvider,
    model: str,
    correct_answer: str = None,
) -> CollectionResult:
    """Collect a response using the double-call protocol.

    Args:
        question: The question text (may include CoT instructions to strip)
        llm_provider: LLM provider for making API calls
        model: Model identifier
        correct_answer: Optional ground truth for scoring

    Returns:
        CollectionResult with verdict and rationale
    """
    total_latency = 0
    total_input = 0
    total_output = 0

    # ==========================================================================
    # CALL A: Verdict (simple answer request)
    # ==========================================================================
    verdict_prompt = make_verdict_prompt(question)
    verdict_response = llm_provider.call(model, verdict_prompt)

    verdict = extract_verdict(verdict_response.text)
    if not verdict:
        raise ValueError(f"Failed to extract verdict from: {verdict_response.text[:100]}")

    total_latency += verdict_response.latency_ms or 0
    total_input += verdict_response.input_tokens or 0
    total_output += verdict_response.output_tokens or 0

    # ==========================================================================
    # CALL B: Rationale (explanation for aggregation)
    # ==========================================================================
    rationale_prompt = make_rationale_prompt(question, verdict)
    rationale_response = llm_provider.call(model, rationale_prompt)

    rationale = rationale_response.text
    if not rationale:
        rationale = "[Rationale unavailable]"

    total_latency += rationale_response.latency_ms or 0
    total_input += rationale_response.input_tokens or 0
    total_output += rationale_response.output_tokens or 0

    # Score if ground truth provided
    is_correct = None
    if correct_answer and verdict:
        is_correct = verdict.upper() == correct_answer.upper()

    return CollectionResult(
        verdict=verdict,
        rationale=rationale,
        is_correct=is_correct,
        latency_ms=total_latency,
        input_tokens=total_input,
        output_tokens=total_output,
    )


# Example usage
if __name__ == "__main__":
    from example import OpenAIProvider

    # Initialize provider with xhigh reasoning for GPT-5.2
    llm = OpenAIProvider(
        api_key="your-api-key",
        reasoning_effort="xhigh",  # Critical for GPT-5.2
    )

    # Example question
    question = """What is 2 + 2?

A) 3
B) 4
C) 5
D) 6

Think through this step-by-step:
1. Analyze what the question is asking
2. Consider each option

ANSWER: [letter]"""

    # Collect with double-call
    result = collect_with_double_call(
        question=question,
        llm_provider=llm,
        model="gpt-5.2",
        correct_answer="B",
    )

    print(f"Verdict: {result.verdict}")
    print(f"Correct: {result.is_correct}")
    print(f"Rationale: {result.rationale[:200]}...")
