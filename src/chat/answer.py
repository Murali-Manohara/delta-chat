"""
Grounded answer: retrieve -> prompt -> generate -> citations.

Grounding is enforced two ways, not just requested in the prompt:
  1. The prompt only ever contains retrieved chunks (never "the whole
     document"), each tagged `[N] <source_ref>: <text>`, and the system
     prompt instructs the model to answer *only* from those, citing
     `[N]` inline, and to say so explicitly if the retrieved context
     doesn't support an answer (refuse/hedge, not hallucinate).
  2. Post-hoc citation resolution: we regex out every `[N]` the model
     used and map it back to the concrete `source_ref` (a PID+location or
     delta entry id) that chunk came from. If the model cites a number
     that wasn't in the retrieved set, that's a groundedness failure we
     can detect and flag (used by eval/metrics.py's groundedness score).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.chat.index import Chunk, RetrievalIndex
from src.chat.llm import LLMClient, LLMResponse
from src.observability.tracing import LLMCall, Trace, estimate_cost

SYSTEM_PROMPT = """You are a grounded assistant answering questions about two \
revisions of an engineering document (PID A = base, PID B = revised) and a \
delta report describing what changed between them.

Rules:
- Answer ONLY using the numbered SOURCES provided below. Do not use outside \
knowledge about compressors, P&IDs, or engineering in general.
- Every factual claim must cite its source inline like [2] using the number \
from the SOURCES list.
- If the sources do not contain enough information to answer, say so plainly \
instead of guessing.
- Be concise and specific (tag numbers, pressures, page numbers) rather than \
generic."""


@dataclass
class Citation:
    marker: str        # "[2]"
    source_ref: str
    doc_label: str
    text: str


@dataclass
class GroundedAnswer:
    text: str
    citations: list[Citation]
    ungrounded_markers: list[str]  # citations the model used but weren't retrieved
    llm_response: LLMResponse


def _build_prompt(question: str, chunks: list[tuple[Chunk, float]]) -> str:
    lines = ["SOURCES:"]
    for i, (chunk, score) in enumerate(chunks, start=1):
        lines.append(f"[{i}] {chunk.doc_label} ({chunk.source_ref}): {chunk.text}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def answer_question(question: str, index: RetrievalIndex, llm: LLMClient,
                     trace: Trace, top_k: int = 8) -> GroundedAnswer:
    with trace.span("retrieve", query=question, top_k=top_k) as span:
        results = index.search(question, top_k=top_k)
        span.attributes["n_retrieved"] = len(results)

    prompt = _build_prompt(question, results)

    with trace.span("llm_call", model=llm.model_name):
        resp = llm.generate(SYSTEM_PROMPT, prompt)

    cost = estimate_cost(resp.model, resp.input_tokens, resp.output_tokens)
    trace.record_llm_call(LLMCall(
        span_name="llm_call", model=resp.model, prompt=prompt, response=resp.text,
        input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
        cost_usd=cost, latency_ms=resp.latency_ms,
    ))

    with trace.span("cite"):
        used_markers = sorted(set(re.findall(r"\[(\d+)\]", resp.text)), key=int)
        citations, ungrounded = [], []
        for m in used_markers:
            idx = int(m) - 1
            if 0 <= idx < len(results):
                chunk, _ = results[idx]
                citations.append(Citation(marker=f"[{m}]", source_ref=chunk.source_ref,
                                           doc_label=chunk.doc_label, text=chunk.text))
            else:
                ungrounded.append(f"[{m}]")

    return GroundedAnswer(text=resp.text, citations=citations,
                           ungrounded_markers=ungrounded, llm_response=resp)
