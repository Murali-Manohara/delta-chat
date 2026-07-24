import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_chat_answer, score_delta


def test_perfect_delta_prediction_scores_1():
    gt = [{"kind": "added", "before": None, "after": "26-PIT-9066 relief valve inlet"}]
    pred = [{"kind": "added", "before": None, "after": "26-PIT-9066 relief valve inlet pressure"}]
    s = score_delta(pred, gt)
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_missed_ground_truth_lowers_recall():
    gt = [
        {"kind": "added", "before": None, "after": "26-PIT-9066 relief valve"},
        {"kind": "removed", "before": "26-TIT-9063 suction gas temperature", "after": None},
    ]
    pred = [{"kind": "added", "before": None, "after": "26-PIT-9066 relief valve"}]
    s = score_delta(pred, gt)
    assert s.recall == 0.5
    assert s.precision == 1.0
    assert len(s.uncovered_gt) == 1


def test_spurious_prediction_lowers_precision():
    gt = [{"kind": "added", "before": None, "after": "26-PIT-9066 relief valve"}]
    pred = [
        {"kind": "added", "before": None, "after": "26-PIT-9066 relief valve"},
        {"kind": "modified", "before": "unrelated old", "after": "unrelated new text about nothing"},
    ]
    s = score_delta(pred, gt)
    assert s.precision == 0.5
    assert s.recall == 1.0
    assert len(s.spurious_pred) == 1


def test_mismatched_kind_does_not_match():
    gt = [{"kind": "added", "before": None, "after": "26-PIT-9066 relief valve"}]
    pred = [{"kind": "removed", "before": "26-PIT-9066 relief valve", "after": None}]
    s = score_delta(pred, gt)
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_grounded_correct_answer_scores_well():
    s = score_chat_answer(
        question="what changed?", answer_text="The HH setpoint changed from 245 to 250 barg [1].",
        citation_refs=["delta:mod_1"], ungrounded_markers=[],
        expected_keywords=["245", "250"],
    )
    assert s.groundedness == 1.0
    assert s.keyword_recall == 1.0
    assert s.correct is True


def test_ungrounded_citation_fails_groundedness():
    s = score_chat_answer(
        question="what changed?", answer_text="Something changed [1] [9].",
        citation_refs=["delta:mod_1"], ungrounded_markers=["[9]"],
        expected_keywords=["245"],
    )
    assert s.groundedness < 1.0
    assert s.correct is False
