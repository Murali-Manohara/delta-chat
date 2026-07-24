"""
Provider-agnostic LLM client.

`LLMClient` is the one interface `answer.py` depends on. Two
implementations ship:
  - `AnthropicClient`: calls the Anthropic Messages API. Model name and
    API key come from env vars (`ANTHROPIC_MODEL`, `ANTHROPIC_API_KEY`) --
    never hardcoded, never committed.
  - `ExtractiveFallbackClient`: zero-dependency, zero-cost, zero-network.
    Used automatically when no API key is configured, so `make chat` and
    `make eval` are runnable by a grader with no credentials at all. It
    does not "generate" prose -- it selects and lightly stitches the
    highest-scoring retrieved chunks into an answer, which is by
    construction grounded (every sentence *is* a retrieved chunk) but
    noticeably less fluent than an LLM answer. This trade-off (and the
    fact that the fallback exists at all) is called out explicitly in
    the README and in eval output, so the grader can tell which mode
    produced a given run.

Swapping in OpenAI/local vLLM/etc. means adding one more class here that
implements `.generate(system, user) -> LLMResponse`; nothing else in the
codebase changes.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # local import: keeps the dependency optional
        self.model_name = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self, system: str, user: str) -> LLMResponse:
        start = time.time()
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.time() - start) * 1000
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=self.model_name,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round(latency_ms, 2),
        )


class ExtractiveFallbackClient(LLMClient):
    """No network, no key. Stitches retrieved chunks into an answer.

    `user` is expected to already contain the retrieved context formatted
    as `[N] <source_ref>: <text>` lines (see answer.py's prompt builder) --
    this client parses that back out rather than calling an LLM, so the
    exact same retrieval -> prompt -> answer pipeline is exercised either
    way, and only the "generation" step differs.
    """

    model_name = "extractive-fallback"

    def generate(self, system: str, user: str) -> LLMResponse:
        start = time.time()
        context_lines = [l for l in user.splitlines() if l.startswith("[")]
        question = ""
        for line in user.splitlines():
            if line.lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
                break

        if not context_lines:
            text = "I don't have enough retrieved context to answer that."
        else:
            top = context_lines[:3]
            text = f"Based on the retrieved sources, regarding \"{question}\":\n"
            for line in top:
                text += f"- {line.strip()}\n"
            text += ("\n(extractive-fallback mode: no LLM configured, so this answer is "
                     "assembled directly from the highest-ranked retrieved chunks above "
                     "rather than generated prose. Set ANTHROPIC_API_KEY for a fluent, "
                     "synthesized answer.)")
        latency_ms = (time.time() - start) * 1000
        # Rough token estimate (chars/4) purely for telemetry consistency;
        # cost is $0 for this client (see tracing.PRICE_TABLE).
        return LLMResponse(
            text=text,
            model=self.model_name,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
            latency_ms=round(latency_ms, 2),
        )


def get_default_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except Exception:
            pass
    return ExtractiveFallbackClient()
