import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.delta.engine import ChangeKind, DeltaEntry, DeltaResult
from src.delta.report import render_markdown, write_report


def _sample_delta():
    entries = [
        DeltaEntry(id="add_1", kind=ChangeKind.ADDED, change_type="note", page=1,
                   location="page 1, region (0,0)-(1,1)", before=None, after="new note",
                   confidence=1.0, block_id_a=None, block_id_b="b1"),
        DeltaEntry(id="mod_1", kind=ChangeKind.MODIFIED, change_type="dimension", page=1,
                   location="page 1, region (0,0)-(1,1)", before="257 barg", after="260 barg",
                   confidence=0.9, block_id_a="a1", block_id_b="b2"),
    ]
    return DeltaResult(pid_a="A", pid_b="B", entries=entries)


def test_markdown_report_contains_summary_and_entries():
    md = render_markdown(_sample_delta())
    assert "1 added, 0 removed, 1 modified" in md
    assert "new note" in md
    assert "257 barg" in md and "260 barg" in md


def test_empty_delta_reports_no_changes():
    empty = DeltaResult(pid_a="A", pid_b="B", entries=[])
    md = render_markdown(empty)
    assert "No meaningful changes" in md


def test_write_report_produces_valid_json(tmp_path):
    paths = write_report(_sample_delta(), str(tmp_path))
    assert os.path.exists(paths["markdown"])
    assert os.path.exists(paths["json"])
    with open(paths["json"]) as f:
        data = json.load(f)
    assert data["counts"] == {"added": 1, "removed": 0, "modified": 1}
    assert len(data["entries"]) == 2
