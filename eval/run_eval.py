#!/usr/bin/env python3
"""
Runnable eval harness: `make eval`.

For every labeled pair in eval/datasets/*_ground_truth.json:
  1. Runs the real pipeline (ingest -> delta) -- same code path as `make run`.
  2. Scores the predicted delta against ground truth (precision/recall/F1).
  3. Builds the retrieval index and answers every QA pair in the matching
     qa_*.json file, scoring groundedness + keyword recall.
Prints one scorecard to stdout and writes eval/results/scorecard.json (and
per-pair delta reports under eval/results/<pair>/) so results are
comparable across runs (regression-friendly, per the requirement).

Native-format pairs (pair_A, pair_B) are the headline P/R/F1 number.
pair_C (scanned) is reported separately since OCR noise makes exact
ground-truth comparison inherently approximate -- see
data/samples/pair_C_cross_document/README.md.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import score_chat_answer, score_delta
from src.chat.answer import answer_question
from src.chat.index import RetrievalIndex
from src.chat.llm import get_default_client
from src.observability.tracing import Trace
from src.pipeline import run_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "eval", "datasets")
RESULTS = os.path.join(ROOT, "eval", "results")

NATIVE_PAIRS = {"pair_A_equipment_schedule", "pair_B_valve_notes"}


def load_ground_truth_files():
    return sorted(glob.glob(os.path.join(DATASETS, "*_ground_truth.json")))


def run_pair_eval(gt_path: str, llm) -> dict:
    with open(gt_path) as f:
        gt = json.load(f)
    pair_id = gt["pair_id"]
    path_a = os.path.join(ROOT, gt["pid_a"]["path"])
    path_b = os.path.join(ROOT, gt["pid_b"]["path"])
    out_dir = os.path.join(RESULTS, pair_id)

    t0 = time.time()
    result = run_pipeline(gt["pid_a"]["pid"], path_a, gt["pid_b"]["pid"], path_b, out_dir)
    delta_predicted = [e.to_dict() for e in result.delta.entries]
    delta_score = score_delta(delta_predicted, gt["expected_deltas"])

    qa_path = os.path.join(DATASETS, f"qa_{pair_id}.json")
    chat_scores = []
    if os.path.exists(qa_path):
        with open(qa_path) as f:
            qa = json.load(f)["qa"]
        index = RetrievalIndex()
        index.add_document(result.doc_a, "PID A")
        index.add_document(result.doc_b, "PID B")
        index.add_delta_report(result.delta)
        index.build()

        for item in qa:
            trace = Trace(kind="eval_chat")
            ans = answer_question(item["question"], index, llm, trace)
            trace.write(runs_dir=os.path.join(ROOT, "runs"))
            cs = score_chat_answer(
                question=item["question"], answer_text=ans.text,
                citation_refs=[c.source_ref for c in ans.citations],
                ungrounded_markers=ans.ungrounded_markers,
                expected_keywords=item["expected_keywords"],
            )
            chat_scores.append(cs.__dict__)

    elapsed = round(time.time() - t0, 2)
    return {
        "pair_id": pair_id,
        "is_native": pair_id in NATIVE_PAIRS,
        "note": gt.get("note"),
        "delta_score": delta_score.__dict__,
        "chat_scores": chat_scores,
        "elapsed_s": elapsed,
    }


def print_scorecard(results: list[dict]):
    print("=" * 78)
    print("DELTA-CHAT EVAL SCORECARD")
    print("=" * 78)

    native_results = [r for r in results if r["is_native"]]
    other_results = [r for r in results if not r["is_native"]]

    def print_delta_row(r):
        ds = r["delta_score"]
        print(f"  {r['pair_id']:32s}  P={ds['precision']:.2f}  R={ds['recall']:.2f}  "
              f"F1={ds['f1']:.2f}   (gt={ds['n_ground_truth']}, pred={ds['n_predicted']}, "
              f"covered={ds['n_covered_gt']})")

    print("\n[Delta P/R/F1 -- native-format pairs (headline)]")
    for r in native_results:
        print_delta_row(r)
    if native_results:
        avg_p = sum(r["delta_score"]["precision"] for r in native_results) / len(native_results)
        avg_r = sum(r["delta_score"]["recall"] for r in native_results) / len(native_results)
        avg_f1 = sum(r["delta_score"]["f1"] for r in native_results) / len(native_results)
        print(f"  {'AVERAGE (native)':32s}  P={avg_p:.2f}  R={avg_r:.2f}  F1={avg_f1:.2f}")

    if other_results:
        print("\n[Delta P/R/F1 -- mixed-format / stress pairs (reported separately, see note)]")
        for r in other_results:
            print_delta_row(r)
            if r.get("note"):
                print(f"      note: {r['note'][:100]}...")

    print("\n[Chat groundedness / keyword-recall]")
    for r in results:
        if not r["chat_scores"]:
            continue
        n = len(r["chat_scores"])
        avg_ground = sum(c["groundedness"] for c in r["chat_scores"]) / n
        avg_kw = sum(c["keyword_recall"] for c in r["chat_scores"]) / n
        n_correct = sum(1 for c in r["chat_scores"] if c["correct"])
        print(f"  {r['pair_id']:32s}  groundedness={avg_ground:.2f}  "
              f"keyword_recall={avg_kw:.2f}  correct={n_correct}/{n}")

    print("\n[Candid failure examples]")
    shown = 0
    for r in results:
        for g in r["delta_score"]["uncovered_gt"][:2]:
            print(f"  MISSED ({r['pair_id']}, {g['kind']}/{g.get('change_type')}): "
                  f"{(g.get('before') or g.get('after') or '')[:90]}")
            shown += 1
        for c in r["chat_scores"]:
            if not c["correct"]:
                print(f"  WEAK CHAT ANSWER ({r['pair_id']}): Q=\"{c['question']}\" "
                      f"kw_recall={c['keyword_recall']} grounded={c['groundedness']}")
                shown += 1
    if shown == 0:
        print("  (none in this run)")

    print("=" * 78)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    llm = get_default_client()
    print(f"[eval running with llm={llm.model_name}]")

    results = [run_pair_eval(gt_path, llm) for gt_path in load_ground_truth_files()]

    print_scorecard(results)

    with open(os.path.join(RESULTS, "scorecard.json"), "w") as f:
        json.dump({"llm_model": llm.model_name, "results": results}, f, indent=2, default=str)
    print(f"\nFull scorecard: {os.path.join(RESULTS, 'scorecard.json')}")


if __name__ == "__main__":
    main()
