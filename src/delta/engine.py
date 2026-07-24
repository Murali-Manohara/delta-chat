"""
Delta engine: turns an AlignmentResult into typed, located, confident
DeltaEntry objects.

Classification rule (deterministic, reproducible -- see README
"Determinism"):
  - unmatched in A only -> REMOVED
  - unmatched in B only -> ADDED
  - matched with score < 99.5 -> MODIFIED (near-100 scores from pass 1
    of alignment are exact text matches on the same page -- true no-ops
    -- and are dropped entirely; they are not part of the delta)
  - matched with score >= 99.5 but text differs by only whitespace/case
    is still classified as unchanged (handled by the >=99.5 cut)

Confidence per entry is NOT the LLM's opinion -- it is a deterministic
function of the alignment score and each block's extraction confidence:

    confidence = alignment_score/100 * min(extraction_conf_a, extraction_conf_b)

For ADDED/REMOVED entries there is no "other side", so confidence is just
the block's own extraction confidence (a low-confidence OCR line that
looks new might just be an OCR miss on the other revision -- we surface
that honestly rather than claiming certainty).

`change_type` (a further sub-classification of *what kind* of content
changed -- dimension vs note vs tag vs generic text) is inferred from the
block's `BlockType` plus a couple of regexes, and is what the delta
report groups by.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum

from src.canonical.model import Block, BlockType, CanonicalDocument
from src.delta.align import AlignmentResult, align

MODIFIED_THRESHOLD = 99.5

_DIM_RE = re.compile(r"\b\d+(\.\d+)?\s?(mm|bar|barg|°c|kw|kg/h|mmscfd|xd|psig?)\b", re.IGNORECASE)
_TAG_RE = re.compile(r"\b\d{2}-[A-Z]{2,4}-\d{3,5}[A-Z]?\b")


class ChangeKind(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


def _infer_change_type(block: Block) -> str:
    if block.block_type == BlockType.TABLE_CELL:
        return "table_cell"
    if _TAG_RE.search(block.text):
        return "tag"
    if _DIM_RE.search(block.text):
        return "dimension"
    if block.block_type == BlockType.NOTE:
        return "note"
    return "text"


@dataclass
class DeltaEntry:
    id: str
    kind: ChangeKind
    change_type: str
    page: int
    location: str            # human-readable "page N, region (x0,y0)-(x1,y1)"
    before: str | None
    after: str | None
    confidence: float
    block_id_a: str | None
    block_id_b: str | None

    def to_dict(self):
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


def _location_str(block: Block) -> str:
    bb = block.bbox.as_tuple()
    return f"page {block.page}, region ({bb[0]},{bb[1]})-({bb[2]},{bb[3]})"


def _row_tag(doc: CanonicalDocument, table_id: str | None, row: int | None) -> str | None:
    """For a table_cell block, find the col=0 ('Tag') cell in the same row
    so a bare cell edit like '245 -> 250 barg' can be reported/cited as
    "26-PIT-9062: 245 -> 250 barg" rather than losing its row identity.
    Without this, a delta entry for a single changed cell is technically
    correct but useless for retrieval/citation ("what changed on
    26-PIT-9062?" wouldn't match a chunk that never mentions the tag)."""
    if table_id is None or row is None:
        return None
    for b in doc.blocks:
        if b.table_id == table_id and b.row == row and b.col == 0:
            return b.text
    return None


def _with_row_context(text: str, tag: str | None) -> str:
    if tag and tag not in text:
        return f"{tag}: {text}"
    return text


@dataclass
class DeltaResult:
    pid_a: str
    pid_b: str
    entries: list[DeltaEntry]

    def counts(self) -> dict:
        out = {"added": 0, "removed": 0, "modified": 0}
        for e in self.entries:
            out[e.kind.value] += 1
        return out

    def to_dict(self):
        return {
            "pid_a": self.pid_a,
            "pid_b": self.pid_b,
            "counts": self.counts(),
            "entries": [e.to_dict() for e in self.entries],
        }


def compute_delta(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                   alignment: AlignmentResult | None = None) -> DeltaResult:
    if alignment is None:
        alignment = align(doc_a, doc_b)

    entries: list[DeltaEntry] = []

    for blk in alignment.unmatched_a:
        tag = _row_tag(doc_a, blk.table_id, blk.row) if blk.block_type == BlockType.TABLE_CELL else None
        entries.append(DeltaEntry(
            id=f"rm_{blk.block_id}",
            kind=ChangeKind.REMOVED,
            change_type=_infer_change_type(blk),
            page=blk.page,
            location=_location_str(blk),
            before=_with_row_context(blk.text, tag),
            after=None,
            confidence=round(blk.extraction_confidence, 3),
            block_id_a=blk.block_id,
            block_id_b=None,
        ))

    for blk in alignment.unmatched_b:
        tag = _row_tag(doc_b, blk.table_id, blk.row) if blk.block_type == BlockType.TABLE_CELL else None
        entries.append(DeltaEntry(
            id=f"add_{blk.block_id}",
            kind=ChangeKind.ADDED,
            change_type=_infer_change_type(blk),
            page=blk.page,
            location=_location_str(blk),
            before=None,
            after=_with_row_context(blk.text, tag),
            confidence=round(blk.extraction_confidence, 3),
            block_id_a=None,
            block_id_b=blk.block_id,
        ))

    for pair in alignment.matched:
        if pair.score >= MODIFIED_THRESHOLD and pair.a.text.strip() == pair.b.text.strip():
            continue  # true no-op, not part of the delta
        if pair.score >= MODIFIED_THRESHOLD:
            continue  # whitespace/case-only difference, treated as unchanged
        conf = round((pair.score / 100.0) * min(pair.a.extraction_confidence, pair.b.extraction_confidence), 3)
        tag = None
        if pair.b.block_type == BlockType.TABLE_CELL:
            tag = _row_tag(doc_b, pair.b.table_id, pair.b.row) or _row_tag(doc_a, pair.a.table_id, pair.a.row)
        entries.append(DeltaEntry(
            id=f"mod_{pair.a.block_id}_{pair.b.block_id}",
            kind=ChangeKind.MODIFIED,
            change_type=_infer_change_type(pair.b),
            page=pair.b.page,
            location=_location_str(pair.b),
            before=_with_row_context(pair.a.text, tag),
            after=_with_row_context(pair.b.text, tag),
            confidence=conf,
            block_id_a=pair.a.block_id,
            block_id_b=pair.b.block_id,
        ))

    entries.sort(key=lambda e: (e.page, e.kind.value))
    return DeltaResult(pid_a=doc_a.meta.pid, pid_b=doc_b.meta.pid, entries=entries)
