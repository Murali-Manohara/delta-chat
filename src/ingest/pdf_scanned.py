"""
Scanned PDF adapter: raster/image-only PDFs with no reliable text layer.

Rasterizes each page (pdf2image / poppler) then runs Tesseract OCR in
`image_to_data` mode, which returns per-word bounding boxes and per-word
confidence -- both of which we carry into the canonical `Block` so
downstream consumers (delta engine, chat citations) can tell the user
"this came from OCR at 62% confidence" rather than presenting OCR output
with the same trust level as a native text layer.

Word-level boxes are grouped into line-level blocks (by `block_num` /
`par_num` / `line_num` from Tesseract's TSV output) to keep block density
comparable to the native adapter's line-level blocks -- this matters for
the delta engine's alignment step, which assumes "one block ~= one line".

`can_handle` is the mirror image of the native adapter's probe: any .pdf
that the native adapter's `_looks_native` rejected is treated as scanned.
Order of registration in the pipeline matters (native is tried first).
"""
from __future__ import annotations

import pytesseract
from pdf2image import convert_from_path

from src.canonical.model import BBox, Block, BlockType, CanonicalDocument, DocumentMeta
from src.ingest.base import FormatAdapter, IngestError, PIDRef, sha256_of
from src.ingest.pdf_native import _looks_native

DPI = 300


class ScannedPdfAdapter(FormatAdapter):
    name = "pdf_scanned"

    def can_handle(self, ref: PIDRef) -> bool:
        return ref.declared_format == "pdf" and not _looks_native(ref.path)

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        try:
            images = convert_from_path(ref.path, dpi=DPI)
        except Exception as e:
            raise IngestError("pdf_scanned.rasterize", f"PID '{ref.pid}': {e}") from e

        blocks: list[Block] = []
        for page_idx, image in enumerate(images, start=1):
            try:
                blocks.extend(self._ocr_page(image, page_idx))
            except Exception as e:
                # Failure on one page must not silently vanish -- caller
                # sees it via the trace's stage errors, but we still try
                # to salvage the rest of the document.
                raise IngestError("pdf_scanned.ocr", f"PID '{ref.pid}' page {page_idx}: {e}") from e

        if not blocks:
            raise IngestError("pdf_scanned", f"PID '{ref.pid}': OCR recovered no text")

        meta = DocumentMeta(
            pid=ref.pid,
            source_path=ref.path,
            format=self.name,
            revision_label=ref.revision_label,
            page_count=len(images),
            sha256=sha256_of(ref.path),
        )
        return CanonicalDocument(meta=meta, blocks=blocks)

    def _ocr_page(self, image, page_idx: int) -> list[Block]:
        # PDF points are at 72 DPI; our raster is at DPI, so scale boxes back.
        scale = 72.0 / DPI
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        lines: dict[tuple, dict] = {}
        n = len(data["text"])
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else 0.0
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            entry = lines.setdefault(key, {"words": [], "confs": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
            entry["words"].append(word)
            entry["confs"].append(conf)
            entry["x0"] = min(entry["x0"], x)
            entry["y0"] = min(entry["y0"], y)
            entry["x1"] = max(entry["x1"], x + w)
            entry["y1"] = max(entry["y1"], y + h)

        blocks = []
        for entry in lines.values():
            text = " ".join(entry["words"]).strip()
            if not text:
                continue
            avg_conf = sum(entry["confs"]) / len(entry["confs"]) / 100.0
            bbox = BBox(
                entry["x0"] * scale, entry["y0"] * scale,
                entry["x1"] * scale, entry["y1"] * scale,
            )
            blocks.append(
                Block(
                    page=page_idx,
                    text=text,
                    bbox=bbox,
                    block_type=BlockType.TEXT,
                    source="pdf_scanned_ocr",
                    extraction_confidence=round(avg_conf, 3),
                )
            )
        return blocks
