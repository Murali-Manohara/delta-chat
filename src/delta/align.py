"""
Alignment: matching content between two revisions.

This is the hard part of the assignment, not the diffing. Two blocks that
say the same thing at slightly different coordinates (a table re-flowed,
a note nudged half an inch) must be recognized as "the same content,
possibly modified" -- not reported as one deletion + one addition. Get
alignment wrong and every delta downstream is noise.

Algorithm (deterministic, no LLM -- see README "Where is the LLM"):
  1. Exact pass: normalize text (casefold, collapse whitespace, strip
     trailing punctuation) and match blocks with identical normalized
     text on the same page + block_type. These are certain non-matches
     for "modified" (if matched, they're unchanged) and are removed from
     the pool immediately -- cheap and correct, no need to fuzzy-score them.
  2. Fuzzy pass on the remainder: candidate pairs are restricted to
     `abs(page_a - page_b) <= PAGE_WINDOW` and the same `block_type`
     (comparing a TAG to a NOTE is never "the same thing modified").
     Score = 0.7 * rapidfuzz.token_sort_ratio + 0.3 * bbox proximity
     score, so identical text that moved far away scores lower than
     identical text that nudged slightly -- this discourages
     spuriously matching two unrelated identical short strings (e.g.
     two blocks that both just say "300#") across the whole document.
  3. Greedy assignment: sort all scored candidate pairs descending,
     accept the highest-scoring pair for each side once, skip anything
     below `MIN_MATCH_SCORE`. Greedy is not globally optimal (a real
     bipartite/Hungarian solve would be), but it's O(n log n) after
     scoring, deterministic, and good enough at this document size --
     documented trade-off, see README "what we cut".
  4. Everything left unmatched in A is a removal candidate; everything
     left unmatched in B is an addition candidate (engine.py turns
     these into typed DeltaEntry objects).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from src.canonical.model import Block, CanonicalDocument

PAGE_WINDOW = 1
MIN_MATCH_SCORE = 55.0
PAGE_SIZE_HINT = 800.0  # points, used to normalize bbox distance


def _normalize(text: str) -> str:
    t = text.casefold().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.\s]+$", "", t)
    return t


def _bbox_center(b: Block) -> tuple[float, float]:
    return ((b.bbox.x0 + b.bbox.x1) / 2, (b.bbox.y0 + b.bbox.y1) / 2)


def _proximity_score(a: Block, b: Block) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return max(0.0, 100.0 * (1 - min(dist / PAGE_SIZE_HINT, 1.0)))


@dataclass
class MatchedPair:
    a: Block
    b: Block
    score: float  # 0-100, combined similarity


@dataclass
class AlignmentResult:
    matched: list[MatchedPair]
    unmatched_a: list[Block]
    unmatched_b: list[Block]


def align(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> AlignmentResult:
    pool_a = {blk.block_id: blk for blk in doc_a.blocks}
    pool_b = {blk.block_id: blk for blk in doc_b.blocks}

    matched: list[MatchedPair] = []

    # --- Pass 1: exact normalized-text match on same page + type ---
    by_key_b: dict[tuple, list[str]] = {}
    for bid, blk in pool_b.items():
        key = (blk.page, blk.block_type, _normalize(blk.text))
        by_key_b.setdefault(key, []).append(bid)

    for aid in list(pool_a.keys()):
        blk_a = pool_a[aid]
        key = (blk_a.page, blk_a.block_type, _normalize(blk_a.text))
        candidates = by_key_b.get(key, [])
        if candidates:
            bid = candidates.pop(0)
            matched.append(MatchedPair(a=blk_a, b=pool_b[bid], score=100.0))
            del pool_a[aid]
            del pool_b[bid]

    # --- Pass 2: fuzzy match on remainder ---
    remaining_a = list(pool_a.values())
    remaining_b = list(pool_b.values())

    scored_pairs = []
    for blk_a in remaining_a:
        for blk_b in remaining_b:
            if blk_a.block_type != blk_b.block_type:
                continue
            if abs(blk_a.page - blk_b.page) > PAGE_WINDOW:
                continue
            text_score = fuzz.token_sort_ratio(_normalize(blk_a.text), _normalize(blk_b.text))
            if text_score < 35:  # cheap short-circuit before bbox math
                continue
            prox_score = _proximity_score(blk_a, blk_b) if blk_a.page == blk_b.page else 0.0
            combined = 0.7 * text_score + 0.3 * prox_score
            if combined >= MIN_MATCH_SCORE:
                scored_pairs.append((combined, blk_a.block_id, blk_b.block_id))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_a, used_b = set(), set()
    a_by_id = {b.block_id: b for b in remaining_a}
    b_by_id = {b.block_id: b for b in remaining_b}

    for score, aid, bid in scored_pairs:
        if aid in used_a or bid in used_b:
            continue
        used_a.add(aid)
        used_b.add(bid)
        matched.append(MatchedPair(a=a_by_id[aid], b=b_by_id[bid], score=score))

    unmatched_a = [b for b in remaining_a if b.block_id not in used_a]
    unmatched_b = [b for b in remaining_b if b.block_id not in used_b]

    return AlignmentResult(matched=matched, unmatched_a=unmatched_a, unmatched_b=unmatched_b)
