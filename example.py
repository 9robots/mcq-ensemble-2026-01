"""
Abstract LLM Provider Interface

Implement this interface to connect the aggregation and debate algorithms
to your LLM infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    model: str
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def call(self, model: str, prompt: str) -> LLMResponse:
        """
        Make a synchronous LLM API call.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            prompt: The prompt to send

        Returns:
            LLMResponse with the model's response text
        """
        pass

    def call_batch(self, requests: list[tuple[str, str]]) -> list[LLMResponse]:
        """
        Make multiple LLM calls. Override for parallel execution.

        Args:
            requests: List of (model, prompt) tuples

        Returns:
            List of LLMResponse objects in same order as requests
        """
        return [self.call(model, prompt) for model, prompt in requests]


# Example implementation using OpenAI
class OpenAIProvider(LLMProvider):
    """Example implementation for OpenAI API."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("pip install openai")

    def call(self, model: str, prompt: str) -> LLMResponse:
        import time
        start = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            model=model,
            latency_ms=latency_ms,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


# Example implementation using Anthropic
class AnthropicProvider(LLMProvider):
    """Example implementation for Anthropic API."""

    def __init__(self, api_key: str):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("pip install anthropic")

    def call(self, model: str, prompt: str) -> LLMResponse:
        import time
        start = time.time()

        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            text=response.content[0].text,
            model=model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
