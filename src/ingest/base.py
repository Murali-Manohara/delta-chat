"""
FormatAdapter: one interface, N formats.

Every adapter takes a `PIDRef` (a resolved pointer to bytes + declared
format) and returns a `CanonicalDocument`. The delta engine and chat layer
depend only on `CanonicalDocument` -- they never import an adapter and
never branch on `.format`. Adding a 4th format (e.g. IFC/BIM, DXF, a CAD
viewer export) means writing one new adapter class; nothing else changes.

`resolve_pid` is intentionally trivial here (it reads a local path). In a
real deployment a PID is a database row: {format, storage_uri, revision
label, page_count, checksum}. We keep that as a dataclass (`PIDRef`) so
swapping the resolver for a real document-management-system client is a
one-file change.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.canonical.model import CanonicalDocument


class IngestError(Exception):
    """Raised when an adapter cannot produce a canonical document.

    Always carries `stage` and `detail` so the caller can put a structured,
    non-swallowed failure into the trace (see observability/tracing.py).
    """

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{stage}] {detail}")


@dataclass
class PIDRef:
    """A resolved pointer to one document revision's bytes + metadata."""
    pid: str
    path: str
    revision_label: str | None = None

    @property
    def declared_format(self) -> str:
        ext = os.path.splitext(self.path)[1].lower()
        return {
            ".pdf": "pdf",       # native vs scanned is detected, not declared
            ".dwg": "dwg",
        }.get(ext, "unknown")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FormatAdapter(ABC):
    """Interface every ingestion adapter implements."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, ref: PIDRef) -> bool:
        """Cheap, format-specific sniff test (extension + content probe)."""
        raise NotImplementedError

    @abstractmethod
    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        """Produce a CanonicalDocument or raise IngestError."""
        raise NotImplementedError


def resolve_pid(pid: str, path: str, revision_label: str | None = None) -> PIDRef:
    if not os.path.exists(path):
        raise IngestError("resolve", f"PID '{pid}' resolves to missing path: {path}")
    return PIDRef(pid=pid, path=path, revision_label=revision_label)


class AdapterRegistry:
    """Tries adapters in registration order; first `can_handle` wins."""

    def __init__(self):
        self._adapters: list[FormatAdapter] = []

    def register(self, adapter: FormatAdapter):
        self._adapters.append(adapter)
        return self

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        for adapter in self._adapters:
            if adapter.can_handle(ref):
                return adapter.ingest(ref)
        raise IngestError("dispatch", f"No adapter could handle PID '{ref.pid}' ({ref.path})")
