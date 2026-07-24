#!/usr/bin/env python3
"""
Grounded chat CLI over PID A, PID B, and the delta report.

    python3 scripts/chat.py --pid-a ... --path-a ... --pid-b ... --path-b ...
    python3 scripts/chat.py                       # uses pair_A sample
    python3 scripts/chat.py -q "what changed near the suction scrubber?"  # single-shot

Runs the full pipeline first (ingest -> delta) so the index always
reflects the current pair, then either drops into an interactive loop or
answers one question and exits (`-q`, used by eval and by CI-style demos
that shouldn't block on stdin).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chat.answer import answer_question
from src.chat.index import RetrievalIndex
from src.chat.llm import get_default_client
from src.observability.tracing import Trace
from src.pipeline import run_pipeline

DEFAULT_A = ("26-KA-901-RevA", "data/samples/pair_A_equipment_schedule/26-KA-901_RevA.pdf")
DEFAULT_B = ("26-KA-901-RevB", "data/samples/pair_A_equipment_schedule/26-KA-901_RevB.pdf")


def build_index(pid_a, path_a, pid_b, path_b, out_dir):
    result = run_pipeline(pid_a, path_a, pid_b, path_b, out_dir)
    index = RetrievalIndex()
    index.add_document(result.doc_a, "PID A")
    index.add_document(result.doc_b, "PID B")
    index.add_delta_report(result.delta)
    index.build()
    return index, result


def ask(index, question, llm):
    trace = Trace(kind="chat")
    ans = answer_question(question, index, llm, trace)
    trace.write()
    print(f"\n> {question}\n")
    print(ans.text)
    print("\nCitations:")
    for c in ans.citations:
        print(f"  {c.marker} -> {c.source_ref} ({c.doc_label})")
    if ans.ungrounded_markers:
        print(f"  WARNING: ungrounded citation markers used: {ans.ungrounded_markers}")
    totals = trace.totals()
    print(f"\n[{ans.llm_response.model} | {totals['total_input_tokens']}in/"
          f"{totals['total_output_tokens']}out tok | ${totals['total_cost_usd']} | "
          f"{totals['total_duration_ms']}ms]")
    return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid-a", default=DEFAULT_A[0])
    p.add_argument("--path-a", default=DEFAULT_A[1])
    p.add_argument("--pid-b", default=DEFAULT_B[0])
    p.add_argument("--path-b", default=DEFAULT_B[1])
    p.add_argument("--out", default="out/latest")
    p.add_argument("-q", "--question", default=None, help="single-shot mode")
    args = p.parse_args()

    index, _ = build_index(args.pid_a, args.path_a, args.pid_b, args.path_b, args.out)
    llm = get_default_client()
    print(f"[chat ready | llm={llm.model_name} | {len(index.chunks)} retrievable chunks]")

    if args.question:
        ask(index, args.question, llm)
        return

    print("Type a question (or 'exit'):")
    while True:
        try:
            q = input("> ")
        except EOFError:
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue
        ask(index, q, llm)


if __name__ == "__main__":
    main()
