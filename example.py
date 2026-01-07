"""
Abstract LLM Provider Interface

Implement this interface to connect the aggregation and debate algorithms
to your LLM infrastructure.

IMPORTANT: Model APIs often default to minimal reasoning. Always verify the
reasoning/thinking parameter matches what vendors used in their benchmarks.
See model_configs.py for exact configurations.
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
    """
    OpenAI API provider with reasoning_effort support.

    CRITICAL: GPT-5.2 defaults to reasoning_effort="none" which gives ~75% on GPQA.
    Setting reasoning_effort="xhigh" achieves vendor-reported 92.4%.

    Available reasoning levels: none|minimal|low|medium|high|xhigh
    """

    def __init__(self, api_key: str, reasoning_effort: str = None):
        """
        Args:
            api_key: OpenAI API key
            reasoning_effort: Reasoning depth for GPT-5.x models.
                              Options: none|minimal|low|medium|high|xhigh
                              GPT-5.2 defaults to "none"; use "xhigh" for benchmark-grade accuracy.
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("pip install openai")
        self.reasoning_effort = reasoning_effort

    def call(self, model: str, prompt: str) -> LLMResponse:
        import time
        start = time.time()

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add reasoning_effort for GPT-5.x models
        # CRITICAL: GPT-5.2 default is "none" (~75%); "xhigh" gets ~92%
        if self.reasoning_effort and model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            # Non-reasoning models use temperature
            kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**kwargs)

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
    """
    Anthropic API provider.

    Note: Claude's extended thinking is enabled automatically for supported models
    (claude-opus-4-5, etc.) - no configuration parameter needed.
    """

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


# Example implementation using Google Gemini (Vertex AI)
class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider via Vertex AI with thinkingLevel support.

    IMPORTANT: Gemini 3 Pro supports thinkingLevel: "low" or "high".
    Default should be "high" but set explicitly for benchmark-grade accuracy.

    Note: Gemini 3 Deep Think (93.8% on GPQA) is a separate model, not a parameter.
    """

    def __init__(self, project_id: str, region: str = "global", thinking_level: str = "HIGH"):
        """
        Args:
            project_id: Google Cloud project ID
            region: Vertex AI region (default: "global")
            thinking_level: Thinking depth for Gemini 3 models.
                           Options: "LOW" or "HIGH" for Pro; "MINIMAL"|"LOW"|"MEDIUM"|"HIGH" for Flash
                           Use "HIGH" for benchmark-grade accuracy.
        """
        try:
            from google import genai
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
            )
        except ImportError:
            raise ImportError("pip install google-genai")
        self.thinking_level = thinking_level

    def call(self, model: str, prompt: str) -> LLMResponse:
        import time
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        start = time.time()

        # Configure thinking for Gemini 3 models
        thinking_cfg = None
        if "gemini-3" in model:
            thinking_cfg = ThinkingConfig(thinking_level=self.thinking_level)

        config = GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            max_output_tokens=16384,
            thinking_config=thinking_cfg,
        )

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            text=response.text,
            model=model,
            latency_ms=latency_ms,
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else None,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else None,
        )
