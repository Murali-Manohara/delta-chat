"""
DWG adapter -- real seam, stubbed implementation.

Per the assignment's own FAQ, DWG is allowed to be "a real stub behind
the adapter seam" rather than a fully working parser. This class *is*
registered in the same `AdapterRegistry` as the working adapters
(`can_handle` correctly claims any `.dwg` file), so the pipeline's
dispatch logic, error handling, and tracing all exercise the real code
path -- it just terminates in a documented, structured `IngestError`
instead of a `CanonicalDocument`.

What a real implementation would do (see README "what we cut" for why we
didn't build this in the time window):
  1. Convert DWG -> DXF via ODA File Converter (free, scriptable) or a
     licensed AutoCAD/Teigha SDK, since DWG itself is a closed binary
     format with no good pure-Python parser.
  2. Parse the DXF with `ezdxf` (pure Python, actively maintained):
     iterate modelspace entities (TEXT, MTEXT, DIMENSION, INSERT/blocks
     for tag bubbles, LWPOLYLINE for pipe runs), map each to a `Block`:
        - TEXT/MTEXT -> BlockType.TEXT or NOTE
        - DIMENSION entities -> BlockType.DIMENSION (measurement + tolerance
          pulled from `dimension.dxf.text` / actual measurement)
        - block INSERTs whose block name matches a tag-bubble pattern ->
          BlockType.TAG
        - everything else (lines, arcs, polylines) -> BlockType.GEOMETRY,
          bbox from entity extents, text = a serialized entity summary
          (layer, entity type, key params) so it's still diffable/citable.
  3. Layers map naturally to something like a `region` hint for citations
     ("changed on layer PIPING-HP, near coordinate (x, y)").
  4. Confidence is 1.0 throughout -- DXF is a lossless vector format, no
     OCR-style uncertainty.
"""
from __future__ import annotations

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter, IngestError, PIDRef


class DwgAdapter(FormatAdapter):
    name = "dwg"

    def can_handle(self, ref: PIDRef) -> bool:
        return ref.declared_format == "dwg"

    def ingest(self, ref: PIDRef) -> CanonicalDocument:
        # Deliberately not implemented -- see module docstring. Raising a
        # structured IngestError (rather than e.g. returning an empty
        # CanonicalDocument) means this failure is visible in the trace,
        # not silently swallowed, per the observability requirement.
        raise IngestError(
            "dwg",
            f"PID '{ref.pid}': DWG adapter is a stub. Real path: DWG->DXF "
            f"(ODA File Converter) -> ezdxf entity walk -> Block per "
            f"TEXT/DIMENSION/INSERT/geometry entity. Not implemented in "
            f"this submission; see src/ingest/dwg.py module docstring.",
        )
