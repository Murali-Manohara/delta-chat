"""
Delta report: human-readable (Markdown) + machine-parseable (JSON)
rendering of a DeltaResult.

The Markdown report is deliberately structured (headers per change_type,
one line per entry with page/location/confidence) so it doubles as a
*retrievable source for chat* -- src/chat/index.py chunks this report the
same way it chunks canonical document blocks, and cites back to entry ids
like `[delta:mod_...]`.
"""
from __future__ import annotations

import json
import os

from src.delta.engine import DeltaResult


def render_markdown(delta: DeltaResult) -> str:
    counts = delta.counts()
    lines = [
        f"# Delta Report: {delta.pid_a} -> {delta.pid_b}",
        "",
        f"**Summary:** {counts['added']} added, {counts['removed']} removed, "
        f"{counts['modified']} modified ({len(delta.entries)} total changes).",
        "",
    ]

    by_type: dict[str, list] = {}
    for e in delta.entries:
        by_type.setdefault(e.change_type, []).append(e)

    for change_type in sorted(by_type):
        group = by_type[change_type]
        lines.append(f"## {change_type} ({len(group)})")
        lines.append("")
        for e in group:
            lines.append(f"### `{e.id}` — {e.kind.value.upper()} — {e.location}")
            if e.before is not None:
                lines.append(f"- **before:** {e.before}")
            if e.after is not None:
                lines.append(f"- **after:** {e.after}")
            lines.append(f"- **confidence:** {e.confidence}")
            lines.append("")

    if not delta.entries:
        lines.append("_No meaningful changes detected between these two revisions._")

    return "\n".join(lines)


def write_report(delta: DeltaResult, out_dir: str, basename: str = "delta_report") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{basename}.md")
    json_path = os.path.join(out_dir, f"{basename}.json")

    with open(md_path, "w") as f:
        f.write(render_markdown(delta))
    with open(json_path, "w") as f:
        json.dump(delta.to_dict(), f, indent=2)

    return {"markdown": md_path, "json": json_path}
