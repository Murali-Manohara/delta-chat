import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.canonical.model import BBox, Block, BlockType, CanonicalDocument, DocumentMeta
from src.delta.engine import ChangeKind, compute_delta


def _doc(pid, blocks):
    return CanonicalDocument(
        meta=DocumentMeta(pid=pid, source_path="x", format="pdf_native", page_count=1),
        blocks=blocks,
    )


def _blk(text, page=1, x=0, y=0, block_type=BlockType.TEXT, table_id=None, row=None, col=None):
    return Block(page=page, text=text, bbox=BBox(x, y, x + 50, y + 10), block_type=block_type,
                 table_id=table_id, row=row, col=col)


def test_identical_documents_produce_no_delta():
    blocks = [_blk("hello world"), _blk("22. design pressure 257 barg", x=100)]
    a = _doc("A", blocks)
    b = _doc("B", [_blk(bl.text, x=bl.bbox.x0) for bl in blocks])
    delta = compute_delta(a, b)
    assert delta.entries == []
    assert delta.counts() == {"added": 0, "removed": 0, "modified": 0}


def test_added_and_removed_are_classified_correctly():
    a = _doc("A", [_blk("kept"), _blk("removed note", x=200)])
    b = _doc("B", [_blk("kept"), _blk("added note", x=500)])
    delta = compute_delta(a, b)
    kinds = {e.kind for e in delta.entries}
    assert ChangeKind.ADDED in kinds
    assert ChangeKind.REMOVED in kinds
    removed = [e for e in delta.entries if e.kind == ChangeKind.REMOVED][0]
    added = [e for e in delta.entries if e.kind == ChangeKind.ADDED][0]
    assert removed.before == "removed note"
    assert added.after == "added note"


def test_edited_setpoint_is_modified_with_partial_confidence():
    a = _doc("A", [_blk("HH setpoint 245 barg", x=100, y=100)])
    b = _doc("B", [_blk("HH setpoint 250 barg", x=100, y=100)])
    delta = compute_delta(a, b)
    assert len(delta.entries) == 1
    e = delta.entries[0]
    assert e.kind == ChangeKind.MODIFIED
    assert e.before == "HH setpoint 245 barg"
    assert e.after == "HH setpoint 250 barg"
    assert 0.0 < e.confidence < 1.0


def test_dimension_change_type_inferred_from_units():
    a = _doc("A", [_blk("Design pressure 257 barg", x=100)])
    b = _doc("B", [_blk("Design pressure 260 barg", x=100)])
    delta = compute_delta(a, b)
    assert delta.entries[0].change_type == "dimension"


def test_tag_change_type_inferred_from_instrument_tag_pattern():
    a = _doc("A", [_blk("26-PIT-9055 reading X", x=100)])
    b = _doc("B", [_blk("26-PIT-9055 reading Y", x=100)])
    delta = compute_delta(a, b)
    assert delta.entries[0].change_type == "tag"


def test_table_cell_modification_gets_row_tag_context():
    a = _doc("A", [
        _blk("26-PIT-9062", x=0, table_id="t1", row=0, col=0, block_type=BlockType.TABLE_CELL),
        _blk("HH: 245 barg", x=100, table_id="t1", row=0, col=1, block_type=BlockType.TABLE_CELL),
    ])
    b = _doc("B", [
        _blk("26-PIT-9062", x=0, table_id="t1", row=0, col=0, block_type=BlockType.TABLE_CELL),
        _blk("HH: 250 barg", x=100, table_id="t1", row=0, col=1, block_type=BlockType.TABLE_CELL),
    ])
    delta = compute_delta(a, b)
    modified = [e for e in delta.entries if e.kind == ChangeKind.MODIFIED]
    assert len(modified) == 1
    assert "26-PIT-9062" in modified[0].before
    assert "26-PIT-9062" in modified[0].after


def test_delta_is_reproducible_across_runs():
    a = _doc("A", [_blk("kept"), _blk("removed note", x=200)])
    b = _doc("B", [_blk("kept"), _blk("added note", x=500)])
    d1 = compute_delta(a, b)
    d2 = compute_delta(a, b)
    ids1 = sorted(e.id for e in d1.entries)
    ids2 = sorted(e.id for e in d2.entries)
    assert ids1 == ids2
