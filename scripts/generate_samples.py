"""
Generates the sample document pairs under data/samples/ AND the matching
eval/datasets/*_ground_truth.json in one script, from a single authored
content structure -- so ground truth is never re-typed by hand from the
rendered PDF (which would risk drifting from what's actually in the
file). See data/samples/*/PROVENANCE.md for how each pair maps to the
real uploaded P&IDs.

Run: python3 scripts/generate_samples.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from pdf2image import convert_from_path
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "data", "samples")
DATASETS = os.path.join(ROOT, "eval", "datasets")

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)


def render_doc(path, title, notes, table_rows, table_header):
    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = [Paragraph(title, h1), Spacer(1, 10)]
    story.append(Paragraph("Instrument &amp; Equipment Schedule", h2))
    data = [table_header] + table_rows
    tbl = Table(data, repeatRows=1, colWidths=[95, 150, 90, 90])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Notes", h2))
    for n in notes:
        story.append(Paragraph(n, body))
        story.append(Spacer(1, 3))
    doc.build(story)


def diff_ground_truth(rows_a, rows_b, notes_a, notes_b, id_col=0):
    """Author-level ground truth: a plain list of {kind, before, after}
    derived directly from the structured content lists we render from --
    independent of anything the delta engine itself computes."""
    gt = []

    a_by_key = {r[id_col]: r for r in rows_a}
    b_by_key = {r[id_col]: r for r in rows_b}
    for key in a_by_key.keys() - b_by_key.keys():
        gt.append({"kind": "removed", "change_type": "table_cell", "before": " | ".join(a_by_key[key]), "after": None})
    for key in b_by_key.keys() - a_by_key.keys():
        gt.append({"kind": "added", "change_type": "table_cell", "before": None, "after": " | ".join(b_by_key[key])})
    for key in a_by_key.keys() & b_by_key.keys():
        if a_by_key[key] != b_by_key[key]:
            gt.append({"kind": "modified", "change_type": "table_cell",
                       "before": " | ".join(a_by_key[key]), "after": " | ".join(b_by_key[key])})

    # Notes are aligned by their leading number ("19.", "22.") rather than
    # treated as an unordered set-difference -- an edited note (same
    # number, different text) is a MODIFIED entry, not a remove+add pair.
    # This matches how the delta engine's alignment step (rightly) treats
    # "the same note, edited" -- see src/delta/align.py docstring.
    import re as _re
    def _note_num(n):
        m = _re.match(r"^\s*(\d+)\.", _strip_tags(n))
        return m.group(1) if m else None

    a_by_num = {_note_num(n): n for n in notes_a if _note_num(n)}
    b_by_num = {_note_num(n): n for n in notes_b if _note_num(n)}
    for num in a_by_num.keys() - b_by_num.keys():
        gt.append({"kind": "removed", "change_type": "note", "before": _strip_tags(a_by_num[num]), "after": None})
    for num in b_by_num.keys() - a_by_num.keys():
        gt.append({"kind": "added", "change_type": "note", "before": None, "after": _strip_tags(b_by_num[num])})
    for num in a_by_num.keys() & b_by_num.keys():
        if a_by_num[num] != b_by_num[num]:
            gt.append({"kind": "modified", "change_type": "note",
                       "before": _strip_tags(a_by_num[num]), "after": _strip_tags(b_by_num[num])})

    return gt


def _strip_tags(s):
    import re
    return re.sub(r"<[^>]+>", "", s)


# ---------------------------------------------------------------------
# PAIR A: "26-KA-901 Lift Gas Compressor — Instrument Schedule"
#   derived from real tag numbers / setpoints / notes present in the
#   uploaded Lift_Gas_compressor-P_ID.pdf.
# ---------------------------------------------------------------------
HEADER_A = ["Tag", "Service", "Setpoints", "Design Press."]

rows_a_A = [
    ["26-PDIT-9054", "Compressor pressurized stop DP", "H: 0.3 / HH: 0.6 barg", "286 barg"],
    ["26-PIT-9062", "Suction scrubber pressure", "LL: 120 / HH: 245 barg", "286 barg"],
    ["26-PIT-9058", "Discharge pressure", "LL: 50 / HH: 140 barg", "286 barg"],
    ["26-TIT-9063", "Suction gas temperature", "monitoring only", "160 C"],
    ["26-PIT-9055", "Primary seal gas suction pressure", "monitoring only", "286 barg"],
    ["26-PDIT-9757", "Balance line cooler DP", "monitoring only", "10 barg (tube rupture)"],
]
rows_a_B = [
    ["26-PDIT-9054", "Compressor pressurized stop DP", "H: 0.3 / HH: 0.6 barg", "286 barg"],
    ["26-PIT-9062", "Suction scrubber pressure", "LL: 120 / HH: 250 barg", "286 barg"],  # MODIFIED HH 245->250
    ["26-PIT-9058", "Discharge pressure", "LL: 50 / HH: 140 barg", "286 barg"],
    # 26-TIT-9063 REMOVED (decommissioned per Rev B)
    ["26-PIT-9055", "Primary seal gas suction pressure", "monitoring only", "286 barg"],
    ["26-PDIT-9757", "Balance line cooler DP", "monitoring only", "10 barg (tube rupture)"],
    ["26-PIT-9066", "Relief valve inlet pressure", "monitoring only", "286 barg"],  # ADDED
]

notes_a_A = [
    "1. 26-PDI-9054 HH initiate pressurized compressor stop.",
    "5. Oil change by using temporary arrangement with hoses.",
    "9. Manual drain prior to each start-up. Push button with permissive for start-up sequence.",
    "19. Suction strainer (commissioning phase only). High alarm at 40% of design DP and compressor high-high trip at 50% of design DP recommended by compressor vendor.",
    "22. Design pressure in external system downstream compressor 257 barg.",
    "30. Safety critical heat tracing - hydrate mitigation (25 C).",
]
notes_a_B = [
    "1. 26-PDI-9054 HH initiate pressurized compressor stop.",
    # note 5 REMOVED: permanent oil-change connection installed, temporary hose arrangement no longer applicable.
    "9. Manual drain prior to each start-up. Push button with permissive for start-up sequence.",
    "19. Suction strainer (commissioning phase only). High alarm at 45% of design DP and compressor high-high trip at 55% of design DP recommended by compressor vendor.",  # MODIFIED 40/50 -> 45/55
    "22. Design pressure in external system downstream compressor 260 barg.",  # MODIFIED 257 -> 260
    "30. Safety critical heat tracing - hydrate mitigation (25 C).",
    "36. Field instrument air filter-regulator added upstream of ESDV per site walkdown.",  # ADDED
]

# ---------------------------------------------------------------------
# PAIR B: "26-KA-902 Export Gas Compressor — Valve & Vent Schedule"
#   derived from real tag numbers / setpoints / notes present in the
#   uploaded Export_Gas_Compressor-P_ID.pdf.
# ---------------------------------------------------------------------
HEADER_B = ["Tag", "Service", "Setpoints", "Design Press."]

rows_b_A = [
    ["26-PSV-9027A", "Compressor discharge relief", "SP 225.4 barg", "286 barg"],
    ["26-PSV-9027B", "Compressor discharge relief (spare)", "SP 225.4 barg", "286 barg"],
    ["26-PDIT-9015", "Compressor pressurized stop DP", "H: 0.7 / HH: 1.2 barg", "286 barg"],
    ["26-PIT-9023", "Suction scrubber pressure", "LL: 110 / HH: 214 barg", "286 barg"],
    ["26-PIT-9019", "Discharge pressure", "LL: 50 / HH: 135 barg", "286 barg"],
]
rows_b_B = [
    ["26-PSV-9027A", "Compressor discharge relief", "SP 225.4 barg", "286 barg"],
    ["26-PSV-9027B", "Compressor discharge relief (spare)", "SP 225.4 barg", "286 barg"],
    ["26-PDIT-9015", "Compressor pressurized stop DP", "H: 0.7 / HH: 1.2 barg", "286 barg"],
    ["26-PIT-9023", "Suction scrubber pressure", "LL: 110 / HH: 220 barg", "286 barg"],  # MODIFIED HH 214->220
    # 26-PIT-9019 REMOVED
    ["26-TIT-9024", "Discharge gas temperature", "SD HH: 150 C", "160 C"],  # ADDED
]

notes_b_A = [
    "1. 26-PDI-9015 HH initiate pressurized compressor stop.",
    "9. Manual drain prior to start-up. Push button with permissive to start compressor.",
    "13. Upstream straight pipe run min. 10xD, downstream min 5xD.",
    "19. Suction strainer (commissioning phase only). High alarm at 40% of design DP and compressor high-high trip at 50% of design DP recommended by compressor vendor.",
    "22. Design pressure / temperature in external system downstream compressor 225 barg / 150 C.",
]
notes_b_B = [
    "1. 26-PDI-9015 HH initiate pressurized compressor stop.",
    "9. Manual drain prior to start-up. Push button with permissive to start compressor.",
    "13. Upstream straight pipe run min. 12xD, downstream min 6xD.",  # MODIFIED 10/5 -> 12/6
    "19. Suction strainer (commissioning phase only). High alarm at 40% of design DP and compressor high-high trip at 50% of design DP recommended by compressor vendor.",
    "22. Design pressure / temperature in external system downstream compressor 230 barg / 150 C.",  # MODIFIED 225->230
    "27. Vibration monitoring probes recalibrated per vendor bulletin VB-2026-014.",  # ADDED
]


def make_pair(dirname, pid_prefix, title, header, rows_a, rows_b, notes_a, notes_b):
    out_dir = os.path.join(SAMPLES, dirname)
    os.makedirs(out_dir, exist_ok=True)
    path_a = os.path.join(out_dir, f"{pid_prefix}_RevA.pdf")
    path_b = os.path.join(out_dir, f"{pid_prefix}_RevB.pdf")
    render_doc(path_a, f"{title} — Rev A", notes_a, rows_a, header)
    render_doc(path_b, f"{title} — Rev B", notes_b, rows_b, header)

    gt = diff_ground_truth(rows_a, rows_b, notes_a, notes_b)
    gt_path = os.path.join(DATASETS, f"{dirname}_ground_truth.json")
    os.makedirs(DATASETS, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump({
            "pair_id": dirname,
            "pid_a": {"pid": f"{pid_prefix}-RevA", "path": os.path.relpath(path_a, ROOT)},
            "pid_b": {"pid": f"{pid_prefix}-RevB", "path": os.path.relpath(path_b, ROOT)},
            "expected_deltas": gt,
        }, f, indent=2)
    print(f"wrote {path_a}, {path_b}, {gt_path} ({len(gt)} expected deltas)")
    return path_a, path_b


def make_scanned_variant(native_path, out_path, dpi=200):
    """Rasterize a native PDF into an image-only PDF (no text layer) to
    simulate a scanned/photographed revision, per assignment FAQ:
    "print-and-scan (or photograph) a page." """
    images = convert_from_path(native_path, dpi=dpi)
    rgb_images = [im.convert("RGB") for im in images]
    rgb_images[0].save(out_path, save_all=True, append_images=rgb_images[1:])
    print(f"wrote scanned variant {out_path}")


if __name__ == "__main__":
    path_a1, path_b1 = make_pair("pair_A_equipment_schedule", "26-KA-901", "Lift Gas Compressor",
                                  HEADER_A, rows_a_A, rows_a_B, notes_a_A, notes_a_B)
    path_a2, path_b2 = make_pair("pair_B_valve_notes", "26-KA-902", "Export Gas Compressor",
                                  HEADER_B, rows_b_A, rows_b_B, notes_b_A, notes_b_B)

    # Pair C: mixed-format demo -- Rev A scanned (image-only), Rev B native.
    # Mirrors the assignment's own example: "a scanned as-built supersedes
    # a drawing." Reuses pair A's Rev A content as the "as-built scan" and
    # pair A's Rev B content as the clean CAD re-issue, so ground truth is
    # already known from pair_A's dataset (OCR noise is the only new
    # variable -- see eval/datasets/pair_C_scanned_ground_truth.json).
    out_dir_c = os.path.join(SAMPLES, "pair_C_cross_document")
    os.makedirs(out_dir_c, exist_ok=True)
    scanned_a_path = os.path.join(out_dir_c, "26-KA-901_RevA_SCANNED.pdf")
    make_scanned_variant(path_a1, scanned_a_path)
    native_b_path = os.path.join(out_dir_c, "26-KA-901_RevB.pdf")
    render_doc(native_b_path, "Lift Gas Compressor — Rev B", notes_a_B, rows_a_B, HEADER_A)

    gt = diff_ground_truth(rows_a_A, rows_a_B, notes_a_A, notes_a_B)
    with open(os.path.join(DATASETS, "pair_C_cross_document_ground_truth.json"), "w") as f:
        json.dump({
            "pair_id": "pair_C_cross_document",
            "pid_a": {"pid": "26-KA-901-RevA-scanned", "path": os.path.relpath(scanned_a_path, ROOT)},
            "pid_b": {"pid": "26-KA-901-RevB-native", "path": os.path.relpath(native_b_path, ROOT)},
            "note": "Rev A is an image-only (scanned) PDF -- OCR introduces noise, so exact-match "
                    "ground truth here is approximate; this pair is used to demonstrate the "
                    "pdf_scanned adapter and is reported separately in eval, not averaged into "
                    "the native-only P/R/F1 headline number.",
            "expected_deltas": gt,
        }, f, indent=2)
    print("done.")
