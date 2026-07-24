"""
Pipeline orchestration: resolve two PIDs -> ingest -> delta -> report.

This is the one place that wires the format adapters, delta engine, and
report renderer together, and the one place that owns a `Trace` for the
non-chat part of a request. `run_pipeline` is called by both the CLI
(`scripts/run_pipeline.py`) and the eval harness, so eval exercises the
exact same code path a real run does (no shortcut/mocked version).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.canonical.model import CanonicalDocument
from src.delta.engine import DeltaResult, compute_delta
from src.delta.report import write_report
from src.ingest.base import AdapterRegistry, IngestError, PIDRef, resolve_pid
from src.ingest.dwg import DwgAdapter
from src.ingest.pdf_native import NativePdfAdapter
from src.ingest.pdf_scanned import ScannedPdfAdapter
from src.observability.logging import get_logger, log_event
from src.observability.tracing import Trace

logger = get_logger("pipeline")


def build_registry() -> AdapterRegistry:
    # Order matters: native's can_handle only claims PDFs with a real text
    # layer, scanned claims the rest, dwg claims .dwg. First match wins.
    return (
        AdapterRegistry()
        .register(NativePdfAdapter())
        .register(ScannedPdfAdapter())
        .register(DwgAdapter())
    )


@dataclass
class PipelineResult:
    doc_a: CanonicalDocument
    doc_b: CanonicalDocument
    delta: DeltaResult
    report_paths: dict
    trace: Trace


def run_pipeline(pid_a: str, path_a: str, pid_b: str, path_b: str,
                  out_dir: str, revision_a: str | None = None,
                  revision_b: str | None = None) -> PipelineResult:
    trace = Trace(kind="pipeline")
    registry = build_registry()

    ref_a = resolve_pid(pid_a, path_a, revision_a)
    ref_b = resolve_pid(pid_b, path_b, revision_b)

    with trace.span("ingest", pid=pid_a, path=path_a) as span:
        try:
            doc_a = registry.ingest(ref_a)
            span.attributes["adapter"] = doc_a.meta.format
            span.attributes["n_blocks"] = len(doc_a.blocks)
        except IngestError as e:
            log_event(logger, "error", "ingest failed", pid=pid_a, stage=e.stage, detail=e.detail)
            raise

    with trace.span("ingest", pid=pid_b, path=path_b) as span:
        try:
            doc_b = registry.ingest(ref_b)
            span.attributes["adapter"] = doc_b.meta.format
            span.attributes["n_blocks"] = len(doc_b.blocks)
        except IngestError as e:
            log_event(logger, "error", "ingest failed", pid=pid_b, stage=e.stage, detail=e.detail)
            raise

    with trace.span("delta", pid_a=pid_a, pid_b=pid_b) as span:
        delta = compute_delta(doc_a, doc_b)
        span.attributes.update(delta.counts())

    with trace.span("report", out_dir=out_dir):
        report_paths = write_report(delta, out_dir)

    trace_path = trace.write()
    log_event(logger, "info", "pipeline complete", pid=pid_a, stage="pipeline",
              trace_path=trace_path, counts=delta.counts())

    return PipelineResult(doc_a=doc_a, doc_b=doc_b, delta=delta,
                           report_paths=report_paths, trace=trace)
