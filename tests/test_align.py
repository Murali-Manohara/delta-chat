import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.canonical.model import BBox, Block, BlockType, CanonicalDocument, DocumentMeta
from src.delta.align import align


def _doc(pid, blocks):
    return CanonicalDocument(
        meta=DocumentMeta(pid=pid, source_path="x", format="pdf_native", page_count=1),
        blocks=blocks,
    )


def _blk(text, page=1, x=0, y=0, block_type=BlockType.TEXT):
    return Block(page=page, text=text, bbox=BBox(x, y, x + 50, y + 10), block_type=block_type)


def test_identical_blocks_align_exactly():
    a = _doc("A", [_blk("hello world")])
    b = _doc("B", [_blk("hello world")])
    result = align(a, b)
    assert len(result.matched) == 1
    assert result.matched[0].score == 100.0
    assert not result.unmatched_a
    assert not result.unmatched_b


def test_added_block_is_unmatched_in_b():
    a = _doc("A", [_blk("hello world")])
    b = _doc("B", [_blk("hello world"), _blk("a brand new note", x=200)])
    result = align(a, b)
    assert len(result.unmatched_b) == 1
    assert result.unmatched_b[0].text == "a brand new note"
    assert not result.unmatched_a


def test_removed_block_is_unmatched_in_a():
    a = _doc("A", [_blk("hello world"), _blk("an old note", x=200)])
    b = _doc("B", [_blk("hello world")])
    result = align(a, b)
    assert len(result.unmatched_a) == 1
    assert result.unmatched_a[0].text == "an old note"


def test_slightly_edited_text_matches_fuzzily_not_as_add_remove():
    a = _doc("A", [_blk("HH setpoint 245 barg", x=100, y=100)])
    b = _doc("B", [_blk("HH setpoint 250 barg", x=100, y=100)])
    result = align(a, b)
    assert len(result.matched) == 1
    assert not result.unmatched_a
    assert not result.unmatched_b
    assert result.matched[0].score < 100.0


def test_different_block_types_never_match():
    a = _doc("A", [_blk("26-PIT-9055", block_type=BlockType.TAG)])
    b = _doc("B", [_blk("26-PIT-9055", block_type=BlockType.TABLE_CELL)])
    result = align(a, b)
    # exact-match pass keys on (page, type, text) so a type mismatch means
    # no match in pass 1; fuzzy pass explicitly skips cross-type pairs too.
    assert not result.matched
    assert len(result.unmatched_a) == 1
    assert len(result.unmatched_b) == 1


def test_far_apart_identical_short_text_does_not_spuriously_match_across_pages():
    a = _doc("A", [_blk("300#", page=1, x=0, y=0)])
    b = _doc("B", [_blk("300#", page=5, x=700, y=700)])
    result = align(a, b)
    # page window is 1, so page 1 vs page 5 can never even be scored
    assert not result.matched
    assert len(result.unmatched_a) == 1
    assert len(result.unmatched_b) == 1
