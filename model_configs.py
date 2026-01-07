"""
Model Configurations for MCQ Ensemble Benchmark

This file documents the exact configuration used for each model in the benchmark.
These settings are critical for reproducing vendor-reported benchmark performance.

IMPORTANT: Model APIs often default to minimal reasoning. Always verify the
reasoning/thinking parameter matches what vendors used in their benchmarks.
"""

# =============================================================================
# PROPRIETARY COHORT (3 models)
# =============================================================================

PROPRIETARY_MODELS = {
    # OpenAI GPT-5.2
    # - reasoning_effort: "xhigh" required for 92.4% on GPQA Diamond
    # - Default is "none" which gives ~75% (17pp lower!)
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "reasoning_effort": "xhigh",  # CRITICAL: none|minimal|low|medium|high|xhigh
        "max_tokens": 100000,
        "notes": "Base model with xhigh reasoning. Pro version is 12x more expensive for only +0.8pp.",
    },

    # Google Gemini 3 Pro
    # - thinkingLevel: "HIGH" for 91.9% on GPQA Diamond
    # - Default should be HIGH but we set explicitly
    "gemini-3-pro": {
        "provider": "vertex_ai",
        "model_id": "gemini-3-pro-preview",
        "region": "global",
        "thinking_level": "HIGH",  # low|high for Pro, minimal|low|medium|high for Flash
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "notes": "Gemini 3 Deep Think (93.8%) is separate model, not just a parameter.",
    },

    # Anthropic Claude Opus 4.5
    # - Extended thinking is automatic, no parameter needed
    # - Via Vertex AI Partner API
    "claude-opus-4-5": {
        "provider": "vertex_ai_partner",
        "model_id": "claude-opus-4-5@20251101",
        "region": "us-east5",
        "max_tokens": 16384,
        "notes": "Extended thinking enabled automatically. 87% on GPQA Diamond.",
    },
}

# =============================================================================
# OPEN SOURCE COHORT (6 models)
# =============================================================================

OSS_MODELS = {
    # DeepSeek R1 - Reasoning model with <think> blocks
    "deepseek-r1": {
        "provider": "vertex_ai",
        "model_id": "deepseek/deepseek-r1",
        "region": "us-central1",
        "notes": "671B MoE reasoning model. Outputs <think>...</think> blocks.",
    },

    # DeepSeek V3.2 - General model
    "deepseek-v3.2": {
        "provider": "vertex_ai",
        "model_id": "deepseek/deepseek-v3-2",
        "region": "us-central1",
        "notes": "671B MoE general model.",
    },

    # Alibaba Qwen3-235B
    "qwen3-235b": {
        "provider": "vertex_ai",
        "model_id": "qwen/qwen3-235b-a22b-instruct-2507-maas",
        "region": "us-south1",
        "notes": "235B MoE instruction-tuned.",
    },

    # Alibaba Qwen3 Next Thinking
    "qwen3-next-thinking": {
        "provider": "vertex_ai",
        "model_id": "qwen/qwen3-next-80b-a3b-thinking-maas",
        "region": "global",
        "notes": "80B thinking/reasoning model variant.",
    },

    # Meta Llama 4
    "llama-4": {
        "provider": "vertex_ai",
        "model_id": "meta/llama-4-maverick-17b-128e-instruct-maas",
        "region": "us-east5",
        "notes": "17B×128E MoE instruction model.",
    },

    # Mistral Medium 3
    "mistral-medium-3": {
        "provider": "vertex_ai",
        "model_id": "mistral-medium-3",
        "region": "us-central1",
        "notes": "Mistral's medium-tier model.",
    },
}

# =============================================================================
# LESSON LEARNED
# =============================================================================
#
# GOTCHA: Never assume API defaults match vendor-reported benchmarks!
#
# Example: GPT-5.2 defaults to reasoning_effort="none" which gives ~75%.
# Vendor reports 92.4% which requires reasoning_effort="xhigh".
# This 17pp gap is NOT a bug - it's a configuration issue.
#
# Always check vendor documentation for:
# 1. What reasoning/thinking mode was used in their benchmarks
# 2. What the API defaults are (often different!)
# 3. Set parameters explicitly even if docs say "default is X"
#
# =============================================================================
