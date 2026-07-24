"""
Native PDF adapter: born-digital PDFs with an extractable text layer.

Uses pdfplumber to pull per-word/per-line text with bounding boxes, plus
table cells where pdfplumber's table detector fires (useful for equipment
schedules / instrument lists, which is most of the information density in
a P&ID-style document).

Detection heuristic (`can_handle`): open the PDF, sample up to the first 3
pages, and check whether pdfplumber recovers a non-trivial amount of text
per page. A scanned PDF (image-only) will return ~0 characters here even
though it "is" a PDF -- that's exactly the signal that routes it to the
`pdf_scanned` adapter instead. This is why adapters are tried in order and
the *scanned* adapter's `can_handle` is the fallback for any .pdf that
fails this probe.
"""
from __future__ import annotations

import pdfplumber

from src.canonical.model import BBox, Block, BlockType, CanonicalDocument, DocumentMeta
from src.ingest.base import FormatAdapter, IngestError, PIDRef, sha256_of

MIN_CHARS_PER_PAGE_FOR_NATIVE = 40


def _looks_native(path: str) -> bool:
    try:
        with pdfplumber.open(path) as pdf:
            sample_pages = pdf.pages[:3]
            if not sample_pages:
                return False
            total_chars = sum(len(p.extract_text() or "") for p in sample_pages)
            return (total_chars / len(sample_pages)) >= MIN_CHARS_PER_PAGE_FOR_NATIVE
    except Exception:
        return False


class NativePdfAdapter(FormatAdapter):
    name = "pdf_native"

    def can_handle(self, ref: PIDRef) -> bool:
        return ref.declared_format == "pdf" and _looks_native(ref.path)

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        try:
            blocks: list[Block] = []
            with pdfplumber.open(ref.path) as pdf:
                page_count = len(pdf.pages)
                for page_idx, page in enumerate(pdf.pages, start=1):
                    blocks.extend(self._page_lines(page, page_idx))
                    blocks.extend(self._page_tables(page, page_idx))
            if not blocks:
                raise IngestError("pdf_native", f"PID '{ref.pid}': no text recovered")
            meta = DocumentMeta(
                pid=ref.pid,
                source_path=ref.path,
                format=self.name,
                revision_label=ref.revision_label,
                page_count=page_count,
                sha256=sha256_of(ref.path),
            )
            return CanonicalDocument(meta=meta, blocks=blocks)
        except IngestError:
            raise
        except Exception as e:
            raise IngestError("pdf_native", f"PID '{ref.pid}': {e}") from e

    def _page_lines(self, page, page_idx: int) -> list[Block]:
        blocks = []
        for line in page.extract_text_lines(strip=True) or []:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            bbox = BBox(line["x0"], line["top"], line["x1"], line["bottom"])
            blocks.append(
                Block(
                    page=page_idx,
                    text=text,
                    bbox=bbox,
                    block_type=BlockType.NOTE if _looks_like_note(text) else BlockType.TEXT,
                    source="pdf_native",
                    extraction_confidence=1.0,
                )
            )
        return blocks

    def _page_tables(self, page, page_idx: int) -> list[Block]:
        blocks = []
        try:
            tables = page.find_tables()
        except Exception:
            return blocks
        for t_idx, table in enumerate(tables):
            table_id = f"p{page_idx}_t{t_idx}"
            rows = table.extract()
            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    if not cell_text or not str(cell_text).strip():
                        continue
                    # cell bbox: fall back to table bbox sliced isn't exact from
                    # pdfplumber's high-level API, so we use the table's bbox
                    # scaled by row/col as an approximation and flag it.
                    x0, top, x1, bottom = table.bbox
                    blocks.append(
                        Block(
                            page=page_idx,
                            text=str(cell_text).strip(),
                            bbox=BBox(x0, top, x1, bottom),
                            block_type=BlockType.TABLE_CELL,
                            source="pdf_native",
                            extraction_confidence=0.9,
                            table_id=table_id,
                            row=r_idx,
                            col=c_idx,
                        )
                    )
        return blocks


def _looks_like_note(text: str) -> bool:
    import re
    return bool(re.match(r"^\d{1,2}[\.\)]\s", text.strip()))
