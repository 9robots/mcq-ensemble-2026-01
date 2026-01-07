"""
Debate Algorithm - Collaborative Truth-Seeking

Reference implementation from 9robots benchmark pipeline.
Multi-round structured debate for disagreement resolution.

Design principles:
- Steelman -> Critique format (understand before dismissing)
- Symmetric accountability (justify both changing AND maintaining)
- Per-model resolution (each model votes - no single judge bias)
- Truth-seeking goal (evaluated on correctness, not agreement)
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from example import LLMProvider, LLMResponse


MAX_DEBATE_ROUNDS = 5
STUCK_THRESHOLD = 2  # Stop if no changes for this many rounds


@dataclass
class DebateResult:
    """Result of the debate process."""
    consensus_reached: bool
    final_answer: Optional[str]
    is_correct: Optional[bool]
    rounds_taken: int
    resolution_method: str  # 'consensus', 'resolution_consensus', 'no_consensus', etc.
    per_model_results: Dict[str, str] = None


def build_debate_prompt(
    question: str,
    positions: Dict[str, List[Dict]],  # answer -> list of {rationale, model (hidden)}
    round_num: int,
    previous_answer: Optional[str] = None,
) -> str:
    """Build the debate prompt using Steelman -> Critique format.

    Key design decisions:
    - Anonymous positions (no model names shown)
    - Steelman requirement (articulate strongest opposing argument)
    - Symmetric accountability (justify both changing AND maintaining)
    - Truth-seeking framing (not consensus-seeking)
    """
    # Format positions anonymously
    position_text = []
    for answer, arguments in sorted(positions.items()):
        n_supporters = len(arguments)
        position_text.append(f"\n[Position {answer} - {n_supporters} supporter{'s' if n_supporters > 1 else ''}]")
        for i, arg in enumerate(arguments, 1):
            rationale = arg['rationale'] if arg['rationale'] else "(no rationale provided)"
            position_text.append(f"  Argument {i}: \"{rationale}\"")

    positions_formatted = "\n".join(position_text)

    previous_section = ""
    if previous_answer:
        previous_section = f"\nYour previous answer was: {previous_answer}\n"

    prompt = f"""COLLABORATIVE TRUTH-SEEKING - Round {round_num}

You are participating in collaborative truth-seeking.
Your goal is NOT to agree with the group.
Your goal is to find the CORRECT answer.
You will be evaluated on whether your final answer is correct, not on whether you agreed with others.

---

QUESTION:
{question}

---

CURRENT POSITIONS:
{positions_formatted}
{previous_section}
---

TASK: Review the positions and respond using this structure:

STEP 1 - STEELMAN:
For each position you disagree with, state the STRONGEST version of their argument.
What is the best case for their answer? Be charitable.

STEP 2 - CRITIQUE:
Identify the specific inferential step or assumption that is invalid.
Cite evidence from the question that contradicts their reasoning.
Be specific - which exact claim is wrong and why?

STEP 3 - DECISION:
State your final answer.

If CHANGING your answer from {previous_answer or 'your initial position'}:
- Which specific argument convinced you?
- What was wrong with your original reasoning?

If MAINTAINING your answer:
- Which counterarguments did you consider?
- Why exactly do they fail? Be specific.

---

Provide your full reasoning, then state:
FINAL ANSWER: [letter]"""

    return prompt


def build_resolution_prompt(
    question: str,
    positions: Dict[str, List[Dict]],
    debate_history: List[str],
) -> str:
    """Build the resolution prompt for per-model voting after debate.

    Each model reviews the full debate and selects a final answer.
    This avoids single-judge bias - all models participate in resolution.
    """
    # Format debate history
    history_text = "\n\n".join([
        f"=== Round {i+1} ===\n{h}"
        for i, h in enumerate(debate_history)
    ])

    # Format final positions
    position_summary = []
    for answer, arguments in sorted(positions.items()):
        n = len(arguments)
        position_summary.append(f"Position {answer}: {n} supporter(s)")
        for i, arg in enumerate(arguments, 1):
            rationale = arg['rationale'][:500] if arg['rationale'] else "(no rationale)"
            position_summary.append(f"  - Argument {i}: {rationale}")

    positions_text = "\n".join(position_summary)

    return f"""DEBATE RESOLUTION

You have participated in a multi-round debate. Now review all arguments and select your final answer.

IMPORTANT:
- Ignore vote counts. The majority is often wrong on hard problems.
- Evaluate REASONING QUALITY only.
- Look for: correct use of evidence, valid inferences, identification of key assumptions.
- The correct answer is the one with the most rigorous argument, not the most supporters.

---

QUESTION:
{question}

---

FINAL POSITIONS:
{positions_text}

---

DEBATE HISTORY:
{history_text}

---

YOUR TASK:
1. Briefly summarize the key arguments for each position.
2. Identify which argument has the soundest logic and why.
3. State your FINAL ANSWER.

FINAL ANSWER: [letter]"""


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the final answer from a response."""
    patterns = [
        r'FINAL\s+ANSWER:\s*\[?([A-D])\]?',
        r'(?:FINAL\s+)?ANSWER:\s*\[?([A-D])\]?',
    ]

    text = response_text.upper()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: last standalone letter A-D
    matches = re.findall(r'\b([A-D])\b', text[-200:])
    if matches:
        return matches[-1]

    return None


def get_positions_from_responses(responses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group responses by answer."""
    positions = defaultdict(list)
    for r in responses:
        answer = r.get('answer')
        if answer:
            positions[answer].append({
                'rationale': r.get('reasoning', ''),
                'model': r.get('model', 'unknown'),
            })
    return dict(positions)


def check_consensus(responses: List[Dict]) -> Tuple[bool, Optional[str]]:
    """Check if all models agree."""
    answers = [r.get('answer') for r in responses if r.get('answer')]
    if not answers:
        return False, None
    counts = Counter(answers)
    if len(counts) == 1:
        return True, answers[0]
    return False, None


def check_stuck(round_answers: List[Dict[str, str]], threshold: int = STUCK_THRESHOLD) -> bool:
    """Check if no one has changed their answer for threshold rounds."""
    if len(round_answers) < threshold:
        return False

    recent = round_answers[-threshold:]
    first = recent[0]
    for r in recent[1:]:
        if r != first:
            return False
    return True


def run_debate(
    question: str,
    native_responses: List[Dict],
    models: List[str],
    llm_provider: LLMProvider,
    correct_answer: str = None,
    max_rounds: int = MAX_DEBATE_ROUNDS,
) -> DebateResult:
    """Run full debate for a prompt until consensus or stuck.

    Args:
        question: The question text
        native_responses: List of dicts with 'model', 'answer', 'reasoning' keys
        models: List of model names to participate in debate
        llm_provider: LLM provider for making API calls
        correct_answer: Optional ground truth for scoring
        max_rounds: Maximum number of debate rounds

    Returns:
        DebateResult with final answer and metadata
    """
    # Check if debate needed (already unanimous)
    is_consensus, consensus_answer = check_consensus(native_responses)
    if is_consensus:
        is_correct = consensus_answer == correct_answer if correct_answer else None
        return DebateResult(
            consensus_reached=True,
            final_answer=consensus_answer,
            is_correct=is_correct,
            rounds_taken=0,
            resolution_method='already_unanimous',
        )

    # Initialize with native responses
    current_responses = [
        {
            'model': r.get('model', 'unknown'),
            'answer': r.get('answer'),
            'reasoning': r.get('reasoning', ''),
        }
        for r in native_responses
    ]

    round_answers_history = []
    debate_history = []

    for round_num in range(1, max_rounds + 1):
        # Check consensus before running round
        is_consensus, consensus_answer = check_consensus(current_responses)
        if is_consensus:
            is_correct = consensus_answer == correct_answer if correct_answer else None
            return DebateResult(
                consensus_reached=True,
                final_answer=consensus_answer,
                is_correct=is_correct,
                rounds_taken=round_num - 1,
                resolution_method='consensus',
            )

        # Track current answers for stuck detection
        current_answer_map = {r['model']: r['answer'] for r in current_responses}
        round_answers_history.append(current_answer_map)

        # Check if stuck - run per-model resolution
        if check_stuck(round_answers_history):
            positions = get_positions_from_responses(current_responses)
            resolution_responses = _run_resolution_round(
                question=question,
                positions=positions,
                debate_history=debate_history,
                models=models,
                llm_provider=llm_provider,
            )

            is_consensus, consensus_answer = check_consensus(resolution_responses)
            resolution_method = 'resolution_consensus' if is_consensus else 'no_consensus'

            is_correct = None
            if consensus_answer and correct_answer:
                is_correct = consensus_answer == correct_answer

            return DebateResult(
                consensus_reached=is_consensus,
                final_answer=consensus_answer if is_consensus else None,
                is_correct=is_correct,
                rounds_taken=round_num,
                resolution_method=resolution_method,
                per_model_results={r['model']: r['answer'] for r in resolution_responses},
            )

        # Run debate round
        positions = get_positions_from_responses(current_responses)
        round_responses = []

        for model in models:
            # Find this model's previous answer
            prev_answer = next(
                (r['answer'] for r in current_responses if r['model'] == model),
                None
            )

            prompt = build_debate_prompt(
                question=question,
                positions=positions,
                round_num=round_num,
                previous_answer=prev_answer,
            )

            response = llm_provider.call(model, prompt)
            answer = extract_answer(response.text)

            round_responses.append({
                'model': model,
                'answer': answer,
                'reasoning': response.text,
            })

        # Track debate history for resolution
        round_summary = "\n".join([
            f"{r['model']}: {r['answer']} - {r['reasoning'][:300]}"
            for r in round_responses
        ])
        debate_history.append(round_summary)

        current_responses = round_responses

    # Max rounds reached - run per-model resolution
    positions = get_positions_from_responses(current_responses)
    resolution_responses = _run_resolution_round(
        question=question,
        positions=positions,
        debate_history=debate_history,
        models=models,
        llm_provider=llm_provider,
    )

    is_consensus, consensus_answer = check_consensus(resolution_responses)
    resolution_method = 'max_rounds_consensus' if is_consensus else 'max_rounds_no_consensus'

    is_correct = None
    if consensus_answer and correct_answer:
        is_correct = consensus_answer == correct_answer

    return DebateResult(
        consensus_reached=is_consensus,
        final_answer=consensus_answer if is_consensus else None,
        is_correct=is_correct,
        rounds_taken=max_rounds,
        resolution_method=resolution_method,
        per_model_results={r['model']: r['answer'] for r in resolution_responses},
    )


def _run_resolution_round(
    question: str,
    positions: Dict[str, List[Dict]],
    debate_history: List[str],
    models: List[str],
    llm_provider: LLMProvider,
) -> List[Dict]:
    """Each model reviews the debate and votes on final answer."""
    resolution_prompt = build_resolution_prompt(question, positions, debate_history)

    resolution_responses = []
    for model in models:
        response = llm_provider.call(model, resolution_prompt)
        answer = extract_answer(response.text)
        resolution_responses.append({
            'model': model,
            'answer': answer,
            'reasoning': response.text,
        })

    return resolution_responses


# Example usage
if __name__ == "__main__":
    from example import OpenAIProvider

    # Initialize provider
    llm = OpenAIProvider(api_key="your-api-key")

    # Example debate
    result = run_debate(
        question="Complex science question...\nA) Option A\nB) Option B\nC) Option C\nD) Option D",
        native_responses=[
            {"model": "gpt-4", "answer": "A", "reasoning": "Because..."},
            {"model": "claude-3", "answer": "B", "reasoning": "Because..."},
            {"model": "gemini", "answer": "A", "reasoning": "Because..."},
        ],
        models=["gpt-4", "claude-3", "gemini"],
        llm_provider=llm,
        correct_answer="B",
    )

    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Final answer: {result.final_answer}")
    print(f"Correct: {result.is_correct}")
    print(f"Rounds: {result.rounds_taken}")
    print(f"Resolution: {result.resolution_method}")
