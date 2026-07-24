#!/usr/bin/env python3
"""
Reproducible run: `make run` / see README.

    python3 scripts/run_pipeline.py \
        --pid-a 26-KA-901-RevA --path-a data/samples/pair_A_equipment_schedule/26-KA-901_RevA.pdf \
        --pid-b 26-KA-901-RevB --path-b data/samples/pair_A_equipment_schedule/26-KA-901_RevB.pdf \
        --out out/pair_A

Defaults to the pair_A sample if no args given, so `make run` with zero
arguments always works out of the box.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline

DEFAULT_A = ("26-KA-901-RevA", "data/samples/pair_A_equipment_schedule/26-KA-901_RevA.pdf")
DEFAULT_B = ("26-KA-901-RevB", "data/samples/pair_A_equipment_schedule/26-KA-901_RevB.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid-a", default=DEFAULT_A[0])
    p.add_argument("--path-a", default=DEFAULT_A[1])
    p.add_argument("--pid-b", default=DEFAULT_B[0])
    p.add_argument("--path-b", default=DEFAULT_B[1])
    p.add_argument("--out", default="out/latest")
    args = p.parse_args()

    result = run_pipeline(args.pid_a, args.path_a, args.pid_b, args.path_b, args.out)

    counts = result.delta.counts()
    print(f"\nIngested PID A ({result.doc_a.meta.format}): {len(result.doc_a.blocks)} blocks")
    print(f"Ingested PID B ({result.doc_b.meta.format}): {len(result.doc_b.blocks)} blocks")
    print(f"Delta: {counts['added']} added, {counts['removed']} removed, {counts['modified']} modified")
    print(f"Report written to: {result.report_paths['markdown']}")
    print(f"                    {result.report_paths['json']}")
    print(f"Trace written to runs/ (request_id={result.trace.request_id})")


if __name__ == "__main__":
    main()
