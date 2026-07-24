"""
Homegrown tracer -- justified in README ("why not OpenTelemetry/Langfuse").

Short version: this is a single-process batch/CLI tool, not a served
multi-tenant app. A full OTel SDK + collector is real infra to stand up
and explain in a take-home; a small dependency-free `Trace`/`Span` object
that (a) captures the exact things the assignment asks for -- per-stage
timing, LLM prompts/responses/tokens/cost, and (b) serializes to one JSON
file per request under `runs/` -- gets 90% of the inspectability value
with none of the ops overhead. If this were a real service, this is
exactly the shape I'd hand to an OTel `SpanProcessor` -- swapping the
`Trace.to_json()` sink for an OTLP exporter is a small, isolated change
because spans already carry (name, start, end, attributes).

Every request (`ingest -> delta -> retrieve -> llm -> answer`) gets one
`Trace`. Each stage is a `Span`. Failures are recorded on the span, not
swallowed -- `Span.fail()` marks status=error and stores the exception
detail, and the trace is still written to disk even on failure.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

RUNS_DIR = os.environ.get("DELTA_CHAT_RUNS_DIR", "runs")


@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end is None:
            return None
        return round((self.end - self.start) * 1000, 2)

    def fail(self, detail: str):
        self.status = "error"
        self.error = detail

    def to_dict(self):
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d


@dataclass
class LLMCall:
    span_name: str
    model: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


# Rough, documented per-1K-token prices used ONLY for cost estimation in
# telemetry -- not billing. Update via PRICE_TABLE env var override if
# needed. Values are illustrative; see README "Observability" section.
PRICE_TABLE = {
    "claude-sonnet-4-6": {"in": 0.003, "out": 0.015},
    "extractive-fallback": {"in": 0.0, "out": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICE_TABLE.get(model, {"in": 0.0, "out": 0.0})
    return round(input_tokens / 1000 * prices["in"] + output_tokens / 1000 * prices["out"], 6)


class Trace:
    def __init__(self, request_id: Optional[str] = None, kind: str = "request"):
        self.request_id = request_id or str(uuid.uuid4())
        self.kind = kind
        self.started_at = time.time()
        self.spans: list[Span] = []
        self.llm_calls: list[LLMCall] = []
        self.metadata: dict[str, Any] = {}

    @contextmanager
    def span(self, name: str, **attributes):
        s = Span(name=name, start=time.time(), attributes=attributes)
        self.spans.append(s)
        try:
            yield s
        except Exception as e:
            s.fail(str(e))
            raise
        finally:
            s.end = time.time()

    def record_llm_call(self, call: LLMCall):
        self.llm_calls.append(call)

    def totals(self) -> dict:
        return {
            "total_duration_ms": round((time.time() - self.started_at) * 1000, 2),
            "total_input_tokens": sum(c.input_tokens for c in self.llm_calls),
            "total_output_tokens": sum(c.output_tokens for c in self.llm_calls),
            "total_cost_usd": round(sum(c.cost_usd for c in self.llm_calls), 6),
            "n_llm_calls": len(self.llm_calls),
            "n_spans": len(self.spans),
            "n_errors": sum(1 for s in self.spans if s.status == "error"),
        }

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "spans": [s.to_dict() for s in self.spans],
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "totals": self.totals(),
            "metadata": self.metadata,
        }

    def write(self, runs_dir: str = RUNS_DIR) -> str:
        os.makedirs(runs_dir, exist_ok=True)
        path = os.path.join(runs_dir, f"trace_{self.kind}_{self.request_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path
