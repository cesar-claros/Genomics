"""
LLM backend abstraction with a HuggingFace local implementation.

Pluggable so a future OpenAI-compatible or Anthropic backend drops in behind
the same `Backend` protocol. The note in container/pyproject.toml about a
future vLLM rollout explicitly anticipates this: vLLM exposes an
OpenAI-compatible API, so an OpenAIBackend(base_url=...) added later will
substitute for HuggingFaceBackend without touching prompt or synthesis code.

Model load is lazy and cached. First call to a given (model_name, dtype)
pulls weights from HuggingFace (~7-8 GB for Phi-3.5-mini-instruct in fp16);
subsequent calls reuse the loaded model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


_DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"


class Backend(Protocol):
    """A backend wraps a single chat-style generate call."""

    name: str

    def generate(self, system: str, user: str, max_new_tokens: int = 1024) -> str:
        ...


@lru_cache(maxsize=2)
def _load_hf(model_name: str, dtype: str):
    """Load tokenizer + model lazily. Cached per (model_name, dtype) pair.

    We deliberately do NOT pass trust_remote_code=True. Phi-3 (and most other
    modern open models) have first-class support in transformers itself
    (Phi3ForCausalLM), and the custom modeling_phi3.py shipped on HuggingFace
    uses an old cache API that breaks on transformers >= 4.45
    (AttributeError: 'DynamicCache' object has no attribute 'seen_tokens').
    Using the native implementation also enables SDPA attention by default.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        device_map="auto",
    )
    return tokenizer, model


@dataclass
class HuggingFaceBackend:
    """Local LLM via the `transformers` library.

    Args:
      model_name: HuggingFace model id (default: Phi-3.5-mini-instruct).
      dtype:      'fp16' (default), 'bf16', or 'fp32'.
    """

    model_name: str = _DEFAULT_MODEL
    dtype: str = "fp16"

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def generate(self, system: str, user: str, max_new_tokens: int = 1024) -> str:
        import torch

        tokenizer, model = _load_hf(self.model_name, self.dtype)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens, not the prompt echo.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def default_backend() -> Backend:
    """Backend chosen via env vars, falling back to Phi-3.5-mini-instruct fp16.

    Env vars:
      REASONING_MODEL: HuggingFace model id
      REASONING_DTYPE: fp16 / bf16 / fp32
    """
    model = os.environ.get("REASONING_MODEL", _DEFAULT_MODEL)
    dtype = os.environ.get("REASONING_DTYPE", "fp16")
    return HuggingFaceBackend(model_name=model, dtype=dtype)
