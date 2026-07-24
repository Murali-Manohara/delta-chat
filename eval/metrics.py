"""
Eval metrics.

Delta metrics (precision / recall / F1):
    Ground truth is authored at "row"/"note" granularity (see
    scripts/generate_samples.py). The delta engine's output is at
    *block* granularity (a table row becomes several table_cell blocks;
    a wrapped note becomes 1-2 line blocks). A naive exact-string match
    between the two would undercount correct detections just because of
    this granularity mismatch -- so matching here is coverage-based:

    - A ground-truth entry is COVERED if at least one predicted entry of
      the same `kind` (added/removed/modified) has token overlap above
      `MATCH_THRESHOLD` with the ground-truth entry's before/after text.
    - A predicted entry is COUNTED AS CORRECT if it covers at least one
      ground-truth entry (same rule, symmetric).

    recall    = covered ground-truth entries / total ground-truth entries
    precision = correct predicted entries / total predicted entries

    This is a deliberate, documented trade-off (see README "Evaluation
    rigor"): it will not catch a system that reports the same real change
    N times as N separate entries (precision would still look fine) --
    a "changes reported" count alongside P/R/F1 is included specifically
    so that failure mode is visible rather than hidden by the metric.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


MATCH_THRESHOLD = 0.35


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _entry_tokens(entry: dict) -> set[str]:
    return _tokens(entry.get("before")) | _tokens(entry.get("after"))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


@dataclass
class DeltaScore:
    precision: float
    recall: float
    f1: float
    n_ground_truth: int
    n_predicted: int
    n_covered_gt: int
    n_correct_pred: int
    uncovered_gt: list[dict]
    spurious_pred: list[dict]


def score_delta(predicted_entries: list[dict], ground_truth_entries: list[dict]) -> DeltaScore:
    gt_tok = [(_entry_tokens(g), g) for g in ground_truth_entries]
    pred_tok = [(_entry_tokens(p), p) for p in predicted_entries]

    covered_gt_idx = set()
    correct_pred_idx = set()

    for gi, (gtoks, g) in enumerate(gt_tok):
        for pi, (ptoks, p) in enumerate(pred_tok):
            if g["kind"] != p["kind"]:
                continue
            if _overlap(gtoks, ptoks) >= MATCH_THRESHOLD:
                covered_gt_idx.add(gi)
                correct_pred_idx.add(pi)

    n_gt = len(gt_tok)
    n_pred = len(pred_tok)
    n_covered = len(covered_gt_idx)
    n_correct = len(correct_pred_idx)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_covered / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    uncovered = [g for gi, (_, g) in enumerate(gt_tok) if gi not in covered_gt_idx]
    spurious = [p for pi, (_, p) in enumerate(pred_tok) if pi not in correct_pred_idx]

    return DeltaScore(
        precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3),
        n_ground_truth=n_gt, n_predicted=n_pred,
        n_covered_gt=n_covered, n_correct_pred=n_correct,
        uncovered_gt=uncovered, spurious_pred=spurious,
    )


# ------------------------- Chat metrics -------------------------

@dataclass
class ChatScore:
    question: str
    answer: str
    groundedness: float          # fraction of cited markers that resolved to a real chunk
    keyword_recall: float        # fraction of expected keywords present in the answer
    correct: bool                # keyword_recall above threshold AND grounded
    citations: list[str]
    ungrounded_markers: list[str]


def score_chat_answer(question: str, answer_text: str, citation_refs: list[str],
                       ungrounded_markers: list[str], expected_keywords: list[str],
                       correctness_threshold: float = 0.5) -> ChatScore:
    groundedness = 1.0 if not ungrounded_markers else max(
        0.0, 1.0 - len(ungrounded_markers) / max(1, len(ungrounded_markers) + len(citation_refs))
    )
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_recall = hits / len(expected_keywords) if expected_keywords else 1.0
    correct = keyword_recall >= correctness_threshold and groundedness >= 0.99 and len(citation_refs) > 0

    return ChatScore(
        question=question, answer=answer_text, groundedness=round(groundedness, 3),
        keyword_recall=round(keyword_recall, 3), correct=correct,
        citations=citation_refs, ungrounded_markers=ungrounded_markers,
    )
