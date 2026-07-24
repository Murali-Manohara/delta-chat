"""
Canonical representation: the format-agnostic intermediate model.

Every ingestion adapter (native PDF, scanned PDF, DWG-stub) normalizes its
source into this model. Nothing downstream (delta engine, chat/retrieval)
ever branches on source format again -- that is the entire point of the
seam.

Design notes (see README "Pipeline design" for the full rationale):
- A document is a flat list of `Block`s. A block is the smallest unit we
  diff and cite: one line/run of text, a table cell, or (future) a
  geometric primitive. Keeping it flat (rather than a nested tree) makes
  alignment across revisions a straightforward matching problem over a
  list, not a tree-edit-distance problem.
- Every block carries a `page` (1-indexed sheet/page number), a `bbox`
  (x0, y0, x1, y1 in PDF points, origin bottom-left) and a `source`
  provenance tag (which adapter produced it, and with what confidence --
  OCR blocks carry OCR confidence, native-text blocks are confidence 1.0).
- `block_id` is deterministic (hash of page+bbox+text) so re-running
  ingestion on the same bytes reproduces the same ids -- this is what
  makes the delta engine's structural output reproducible (a hard
  requirement).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class BlockType(str, Enum):
    TEXT = "text"
    NOTE = "note"
    TABLE_CELL = "table_cell"
    DIMENSION = "dimension"
    TAG = "tag"  # instrument/equipment tag e.g. "26-PIT-9055"
    TITLE_BLOCK = "title_block"
    GEOMETRY = "geometry"  # stub for DWG entities


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self):
        return (round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1))


@dataclass
class Block:
    page: int
    text: str
    bbox: BBox
    block_type: BlockType = BlockType.TEXT
    source: str = "unknown"          # adapter name, e.g. "pdf_native"
    extraction_confidence: float = 1.0  # 1.0 for native text, OCR conf for scans
    table_id: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None
    block_id: str = field(default="")

    def __post_init__(self):
        if not self.block_id:
            key = f"{self.page}|{self.bbox.as_tuple()}|{self.text.strip()}"
            self.block_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def to_dict(self):
        d = asdict(self)
        d["block_type"] = self.block_type.value
        return d


@dataclass
class DocumentMeta:
    pid: str                     # the document/revision identifier (see FAQ)
    source_path: str
    format: str                  # "pdf_native" | "pdf_scanned" | "dwg"
    revision_label: Optional[str] = None
    page_count: int = 0
    sha256: str = ""


@dataclass
class CanonicalDocument:
    meta: DocumentMeta
    blocks: list[Block] = field(default_factory=list)

    def blocks_on_page(self, page: int):
        return [b for b in self.blocks if b.page == page]

    def to_dict(self):
        return {
            "meta": asdict(self.meta),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @staticmethod
    def from_dict(d: dict) -> "CanonicalDocument":
        meta = DocumentMeta(**d["meta"])
        blocks = []
        for bd in d["blocks"]:
            bbox = BBox(**bd["bbox"])
            bd = dict(bd)
            bd["bbox"] = bbox
            bd["block_type"] = BlockType(bd["block_type"])
            blocks.append(Block(**bd))
        return CanonicalDocument(meta=meta, blocks=blocks)
