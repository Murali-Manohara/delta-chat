# DEMO

Real captured output from this repo, `make run` / `make chat` / `make eval`,
run with **no API key set** (extractive-fallback chat backend — see README
"Where is the LLM"). Regenerate any of it yourself with the commands shown.
The configured LLM provider for this submission is **Groq (Llama 3.3 70B)**
— set `GROQ_API_KEY` in `.env` to see real LLM-generated answers instead of
the fallback text below; `ANTHROPIC_API_KEY` also works as a second
reference provider (see README "Where is the LLM").

## 1. Delta run

```
$ make run
```
```
Ingested PID A (pdf_native): 45 blocks
Ingested PID B (pdf_native): 45 blocks
Delta: 2 added, 3 removed, 8 modified
Report written to: out/pair_A/delta_report.md
                    out/pair_A/delta_report.json
Trace written to runs/ (request_id=dea28572-...)
```

Excerpt of `out/pair_A/delta_report.md`:

```markdown
# Delta Report: 26-KA-901-RevA -> 26-KA-901-RevB

**Summary:** 2 added, 3 removed, 8 modified (13 total changes).

## dimension (1)

### `mod_2456adb30218_81ff9e0b8e7f` — MODIFIED — page 1, region (78.0,360.1)-(375.1,369.1)
- **before:** 22. Design pressure in external system downstream compressor 257 barg.
- **after:** 22. Design pressure in external system downstream compressor 260 barg.
- **confidence:** 0.974

## note (3)

### `add_2eabf9adbd1c` — ADDED — page 1, region (78.0,390.1)-(408.6,399.1)
- **after:** 36. Field instrument air filter-regulator added upstream of ESDV per site walkdown.
- **confidence:** 1.0

### `rm_ceb1b052734b` — REMOVED — page 1, region (78.0,318.1)-(311.1,327.1)
- **before:** 5. Oil change by using temporary arrangement with hoses.
- **confidence:** 1.0

## table_cell (5)

### `mod_c722f34974ba_3d5b4ae48d37` — MODIFIED — page 1, region (93.5,123.2)-(518.5,249.2)
- **before:** 26-PIT-9062: LL: 120 / HH: 245 barg
- **after:** 26-PIT-9062: LL: 120 / HH: 250 barg
- **confidence:** 0.871
```

All 13 entries (plus `tag`/`text` sections) are in the full file; every one
carries a page, a bbox region, before/after, and a confidence score.

## 2. Grounded chat exchange

```
$ make chat Q="Was any instrument removed, and what was its design pressure?"
```
```
[chat ready | llm=extractive-fallback | 103 retrievable chunks]

> Was any instrument removed, and what was its design pressure?

Based on the retrieved sources, regarding "Was any instrument removed, and what was its design pressure?":
- [1] PID A (PID A:page1:2456adb30218): 22. Design pressure in external system downstream compressor 257 barg.
- [2] PID B (PID B:page1:81ff9e0b8e7f): 22. Design pressure in external system downstream compressor 260 barg.
- [3] Delta Report (delta:mod_2456adb30218_81ff9e0b8e7f): [modified/dimension] page 1, region (78.0,360.1)-(375.1,369.1). Before: 22. Design pressure in external system downstream compressor 257 barg.. After: 22. Design pressure in external system downstream compressor 260 barg..

(extractive-fallback mode: no LLM configured, so this answer is assembled directly from the highest-ranked retrieved chunks above rather than generated prose. Set GROQ_API_KEY (or ANTHROPIC_API_KEY) for a fluent, synthesized answer.)

Citations:
  [1] -> PID A:page1:2456adb30218 (PID A)
  [2] -> PID B:page1:81ff9e0b8e7f (PID B)
  [3] -> delta:mod_2456adb30218_81ff9e0b8e7f (Delta Report)

[extractive-fallback | 220in/211out tok | $0.0 | 1.81ms]
```

Every citation resolves to a real chunk (`ungrounded_markers` is empty —
you can check this yourself in `runs/trace_chat_*.json`, which has the
full prompt, the model's raw response, and the per-stage timings for this
exact exchange).

**An honest weaker example**, deliberately included rather than
cherry-picked around: `make chat Q="What changed on 26-PIT-9062 between Rev A and Rev B?"`
retrieves the right chunks (a `26-PIT-9062` mention from PID A and PID B)
but *not* the delta-report entry that actually states the 245→250 barg
change, because the query text alone doesn't share enough tokens with
that entry's text. Groundedness is still 1.0 (nothing hallucinated) but
the answer is unhelpfully terse. This is the retrieval-recall failure
mode documented in README "Evaluation rigor" and "What I'd do next"
(item 3, hybrid retrieval) — not hidden.

## 3. Eval scorecard

```
$ make eval
```
```
==============================================================================
DELTA-CHAT EVAL SCORECARD
==============================================================================

[Delta P/R/F1 -- native-format pairs (headline)]
  pair_A_equipment_schedule         P=0.92  R=1.00  F1=0.96   (gt=7, pred=13, covered=7)
  pair_B_valve_notes                P=0.82  R=0.83  F1=0.83   (gt=6, pred=11, covered=5)
  AVERAGE (native)                  P=0.87  R=0.92  F1=0.89

[Delta P/R/F1 -- mixed-format / stress pairs (reported separately, see note)]
  pair_C_cross_document             P=0.46  R=0.86  F1=0.60   (gt=7, pred=41, covered=6)
      note: Rev A is an image-only (scanned) PDF -- OCR introduces noise, so exact-match ground truth here is ap...

[Chat groundedness / keyword-recall]
  pair_A_equipment_schedule         groundedness=1.00  keyword_recall=0.50  correct=3/6
  pair_B_valve_notes                groundedness=1.00  keyword_recall=0.50  correct=2/4

[Candid failure examples]
  WEAK CHAT ANSWER (pair_A_equipment_schedule): Q="What changed on 26-PIT-9062 between Rev A and Rev B?" kw_recall=0.0 grounded=1.0
  WEAK CHAT ANSWER (pair_A_equipment_schedule): Q="Was any instrument removed in Rev B?" kw_recall=0.0 grounded=1.0
  WEAK CHAT ANSWER (pair_A_equipment_schedule): Q="Was any instrument added in Rev B?" kw_recall=0.0 grounded=1.0
  MISSED (pair_B_valve_notes, removed/table_cell): 26-PIT-9019 | Discharge pressure | LL: 50 / HH: 135 barg | 286 barg
  WEAK CHAT ANSWER (pair_B_valve_notes): Q="What is the setpoint of PSV 26-PSV-9027A?" kw_recall=0.0 grounded=1.0
  WEAK CHAT ANSWER (pair_B_valve_notes): Q="Was any instrument removed between Rev A and Rev B?" kw_recall=0.0 grounded=1.0
  MISSED (pair_C_cross_document, modified/note): 22. Design pressure in external system downstream compressor 257 barg.
==============================================================================

Full scorecard: eval/results/scorecard.json
```

Full machine-readable scorecard (every entry, every chat answer,
per-question scores) is written to `eval/results/scorecard.json` on each
run; a snapshot from this exact run is committed at
`eval/sample_scorecard.json` for reference without re-running anything.

## 4. Scanned-PDF ingestion (format B, real)

```
$ make run-pair-c-scanned
```
```
Ingested PID A (pdf_scanned): ... blocks   <- OCR path, per-word confidence carried into each Block
Ingested PID B (pdf_native): 45 blocks
Delta: 30 added, 7 removed, 4 modified
```

This is the same `run_pipeline.py` / `FormatAdapter` code path as pair_A —
no special-casing — the only difference is which adapter's `can_handle`
claimed the file. See README "Evaluation rigor" for why this pair's
higher change-count is a known OCR-noise artifact, not a hidden bug.
