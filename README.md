# delta-chat

**Document Delta & Grounded Chat** — Applied AI Engineer take-home submission.

Given two revisions of an engineering document (PID A, PID B), this ingests
both regardless of format, computes a structured delta, renders a delta
report, and answers questions grounded in both documents + the delta report,
with citations.

## Quick start

```bash
pip install -r requirements.txt --break-system-packages   # or: make install
make run          # ingest pair_A, produce a delta report -> out/pair_A/
make chat         # interactive grounded chat over pair_A
make chat Q="What changed on 26-PIT-9062?"   # single-shot
make eval         # full eval harness -> scorecard printed + eval/results/scorecard.json
make test         # unit tests (22 tests, no external deps)
```

No API key is required for any of the above — see "Where is the LLM"
below. Set `GROQ_API_KEY` (copy `.env.example` → `.env`) for fluent,
LLM-generated chat answers (Llama 3.3 70B via Groq) instead of the
extractive fallback. `ANTHROPIC_API_KEY` also works as a second reference
provider — see "Where is the LLM" for how provider selection works.

## Sample data — read this first

The two PDFs I was actually handed for this assignment (`26-KA-901` "Lift
Gas Compressor" and `26-KA-902` "Export Gas Compressor") are **two
different pieces of equipment**, not two revisions of one document. A
real revision pair is required to evaluate a delta engine honestly, so
per the assignment's own FAQ ("synthesize them... document their
provenance") I generated three revision pairs from the real tag numbers,
setpoints, and note text in those two PDFs. Full explanation and the
originals are in `data/samples/PROVENANCE.md` and
`data/samples/_source_real_pids/`.

| Pair | Formats | What it exercises |
|---|---|---|
| `pair_A_equipment_schedule` | native ↔ native | primary eval pair |
| `pair_B_valve_notes` | native ↔ native | second, independent eval pair |
| `pair_C_cross_document` | **scanned** ↔ native | OCR ingestion adapter; "a scan supersedes a drawing" |

## What's implemented vs. cut

**Core (A–D), built and working end-to-end:**
- Format-agnostic ingestion: native PDF (pdfplumber) + scanned PDF (Tesseract
  OCR via pdf2image), both real. DWG is a **real stub** behind the same
  interface (`FormatAdapter.can_handle` correctly claims `.dwg`; `ingest`
  raises a structured, documented `IngestError` — see
  `src/ingest/dwg.py` module docstring for exactly what a real
  DWG→DXF→ezdxf implementation would do).
- Delta engine: two-pass deterministic alignment (exact-text, then fuzzy
  text+position) → typed/located/confidence-scored add/remove/modify.
- Delta report: Markdown + JSON, also indexed as a retrievable chat source.
- Grounded chat: TF-IDF retrieval over PID A + PID B + delta report,
  provider-agnostic LLM client, citations resolved and verified against
  what was actually retrieved.

**Required cross-cutting, built:**
- Tracing (`src/observability/tracing.py`): per-request `Trace` with
  per-stage `Span` timing, LLM token/cost telemetry, written to
  `runs/*.json`.
- Structured JSON logs with a correlation id
  (`src/observability/logging.py`).
- Eval harness (`eval/run_eval.py`, `eval/metrics.py`): delta P/R/F1,
  chat groundedness + keyword-recall, runnable via `make eval`,
  regression-comparable (`eval/results/scorecard.json`).

**Bonus (E), cut:** delta markup overlay. This was the first thing cut
under time pressure — see "What I'd do next" — because A–D plus real
observability and a runnable eval harness were explicitly weighted far
higher in the rubric (bonus is capped at +8/100 vs. 20% for eval rigor
alone), and a half-working image-overlay feature is worse signal than a
clearly-scoped cut.

**Also cut / simplified, on purpose:**
- **docker-compose**: `make` targets + a `requirements.txt` cover
  "reproducible run" without the extra surface area of containerizing a
  Python CLI tool for a take-home.
- **Hybrid/embedding retrieval**: TF-IDF only (see `src/chat/index.py`
  docstring for why this is actually a reasonable fit here, not just a
  shortcut) — a real "what changed near the pump?" paraphrase query
  (no literal token match) would need embeddings; flagged as a concrete
  next step below.
- **Full bipartite (Hungarian) alignment**: greedy assignment instead —
  O(n log n), deterministic, good enough at this document size; see
  `src/delta/align.py` docstring.
- **Table-cell bounding boxes**: pdfplumber's high-level table API gives
  a table's overall bbox, not per-cell boxes, so table-cell delta entries
  currently cite the whole table's region rather than the exact cell.
  Noted as a known imprecision, not silently accepted — see
  `src/ingest/pdf_native.py::_page_tables`.

## Where is the LLM (and where it isn't)

The delta engine is **100% deterministic, no LLM** — alignment is fuzzy
text+position matching (rapidfuzz), classification is regex/type-based.
This was a deliberate choice, not a default: an LLM asked "did anything
change between these two blocks of text" is (a) non-deterministic run to
run, which directly violates the assignment's determinism requirement,
and (b) worse at *exhaustive* structural comparison than a
similarity/matching algorithm — LLMs are good at judgment on ambiguous
cases, not at guaranteeing they scanned every block. Where an LLM *is*
used is exactly where judgment-over-retrieved-evidence is the actual
task: answering a natural-language question grounded in retrieved
chunks (`src/chat/answer.py`). That's also the only place in the
pipeline with real non-determinism, and it's isolated there by design —
ingestion and delta re-runs are byte-for-byte reproducible
(`tests/test_engine.py::test_delta_is_reproducible_across_runs`).

Chat has three swappable backends behind one `LLMClient` interface
(`src/chat/llm.py`):
- `GroqClient` — real API call to Groq's OpenAI-compatible endpoint,
  defaulting to Llama 3.3 70B (`llama-3.3-70b-versatile`). This is the
  provider configured for this submission; used automatically if
  `GROQ_API_KEY` is set (or force it with `LLM_PROVIDER=groq`).
- `AnthropicClient` — real API call to Claude, used if `ANTHROPIC_API_KEY`
  is set instead. Kept as a second working implementation specifically
  to demonstrate the `LLMClient` interface is genuinely provider-agnostic
  rather than shaped around Groq's or Anthropic's SDK.
- `ExtractiveFallbackClient` — zero-network, zero-cost, stitches the
  top retrieved chunks into an answer. This is **not** a mocked/fake
  mode for demo purposes — it's what actually runs in `make eval` and
  `make chat` by default, so a grader with no API key can run
  *everything* in this repo, including the eval harness, and see real
  (if less fluent) grounded answers with real citations. The scorecard
  and every trace file record which mode produced a given answer.

Provider selection (`get_default_client()` in `src/chat/llm.py`): explicit
`LLM_PROVIDER=groq|anthropic` wins if set; otherwise whichever `*_API_KEY`
is present is used (Groq checked first); otherwise the extractive
fallback. Any failure constructing a real client (bad key, unreachable
network) falls through to the extractive client rather than crashing the
pipeline.

## Reference architecture

```
 PID A, PID B
     |
     v
 FormatAdapter (pdf_native | pdf_scanned | dwg-stub)  --one interface, N formats--
     |
     v
 CanonicalDocument (Block: page, bbox, type, text, extraction_confidence)
     |
     +---------------------------+
     v                           v
 Delta engine                Retrieval index (TF-IDF)
 align -> classify -> DeltaResult   PID A + PID B + Delta report chunks
     |                           |
     v                           v
 Delta report (MD+JSON)  --->  Grounded chat (LLMClient, swappable)
                                  |
                                  v
                          Answer + citations (verified against retrieved set)

 Observability (Trace/Span, structured logs) wraps every stage above.
 Eval harness (eval/run_eval.py) drives the same pipeline against
 labeled pairs and scores delta P/R/F1 + chat groundedness/correctness.
```

## Repo layout

```
src/
  canonical/model.py     the format-agnostic Block/CanonicalDocument model
  ingest/                base.py (interface+registry), pdf_native.py, pdf_scanned.py, dwg.py
  delta/                 align.py, engine.py, report.py
  chat/                  index.py (retrieval), llm.py (provider-agnostic), answer.py
  observability/         tracing.py, logging.py
  pipeline.py            wires ingest -> delta -> report for one request
eval/
  datasets/               *_ground_truth.json (delta), qa_*.json (chat)
  metrics.py, run_eval.py
data/samples/             3 pairs + PROVENANCE.md + the real uploaded PDFs
scripts/                  generate_samples.py, run_pipeline.py, chat.py
tests/                    22 unit tests, `make test`
```

## Design decisions worth defending

1. **Canonical `Block` is flat, not a tree.** A page's content is a list of
   `Block`s (line, table cell, future geometry), each independently
   diffable and citable. This makes alignment a list-matching problem
   (see `align.py`), not tree-edit-distance, and keeps the delta engine
   and chat retrieval agnostic to source format — a DXF entity walk would
   produce the same flat list shape.
2. **Confidence is compositional, not vibes.** For a `MODIFIED` entry,
   `confidence = alignment_score/100 * min(extraction_confidence of both
   sides)` — a low-confidence OCR match on one side pulls the delta
   entry's confidence down even if the text similarity is high. This is
   what lets `pair_C_cross_document`'s OCR-derived changes honestly show
   lower confidence than `pair_A`'s clean native-PDF changes in the same
   report.
3. **Grounding is enforced structurally, not just prompted.** Every chat
   answer's `[N]` citations are regex-extracted and matched back against
   the actual retrieved chunk list; a citation number the model
   hallucinated (out of range) is flagged as `ungrounded_markers`, not
   silently accepted — this is what `eval/metrics.py::score_chat_answer`
   scores as groundedness, and it works identically whether the backend
   is the real LLM or the extractive fallback.

## Evaluation rigor / honest results

Run `make eval` for a live scorecard; a snapshot is committed at
`eval/sample_scorecard.json`. Headline numbers from that run
(`extractive-fallback` chat backend, no API key):

- **Delta, native-format pairs (pair_A, pair_B):** precision ≈0.87,
  recall ≈0.92, F1 ≈0.89 (coverage-based matching against
  row/note-granularity ground truth — see `eval/metrics.py` docstring
  for exactly how, and why exact-string matching would be the wrong
  metric given the block/row granularity mismatch).
- **Delta, scanned↔native pair (pair_C):** F1 ≈0.60, reported
  *separately*, not averaged into the headline number — OCR noise on
  Rev A produces some spurious near-duplicate line splits that inflate
  the predicted-entry count. This is the single biggest known weakness
  in the system and I'd fix it before anything else with more time (OCR
  post-processing: merge adjacent low-confidence lines before alignment,
  not after).
- **Chat groundedness:** 1.00 on both native pairs (zero hallucinated
  citation numbers across the QA set). **Chat keyword-recall in
  extractive-fallback mode is weak (~0.4)** — the fallback client
  literally pastes the top-3 retrieved chunks rather than synthesizing
  an answer, so it sometimes surfaces the *right* chunk (e.g. a table
  row containing "26-PIT-9062") without restating the specific number a
  keyword check looks for. Groundedness (are the citations real) stays
  perfect either way; fluency/completeness is the known gap of running
  without an LLM key, exactly as documented in `src/chat/llm.py`. With
  `GROQ_API_KEY` set, the same retrieved context goes through Llama 3.3
  70B and keyword-recall should track answer quality more directly — I
  did not have a live key in the sandbox I built this in (Groq's API
  endpoint isn't reachable from that environment's network egress
  allowlist), so I'm reporting the fallback numbers rather than an
  unverified claim about the LLM path. `GroqClient` is unit-tested for
  correct wiring (`tests/test_llm_provider.py`) but not for a live
  response — worth a real smoke test (`make chat Q="..."` with your key
  set) before relying on the LLM-mode numbers.

Candid failure examples (also printed by `make eval`):
- A `pair_B` removed instrument (`26-PIT-9019`) is missed by the delta
  engine — its table cells partially fuzzy-matched against an unrelated
  nearby row instead of being left unmatched, because the greedy
  alignment pass has no global optimality guarantee (documented
  trade-off in `align.py`). A Hungarian-algorithm alignment would likely
  fix this specific case.
- `pair_C`'s OCR pass splits one wrapped note into two lines with
  overlapping text, which the delta engine reports as two smaller
  `modified` entries instead of one — technically not wrong, but noisier
  than the native-PDF version of the same content.

## What I'd do next with more time

1. Fix the OCR line-merging issue above (biggest correctness win for the
   least effort — a post-OCR line-stitching pass keyed on vertical
   proximity + incomplete sentence detection).
2. Swap greedy alignment for a proper assignment solve (`scipy.optimize.
   linear_sum_assignment`) on the fuzzy-candidate score matrix — bounded,
   still fast at this scale, fixes the missed-instrument failure above.
3. Hybridize retrieval: keep TF-IDF for exact tags/numbers, add embedding
   similarity for paraphrase queries, take the max/union.
4. Implement the DWG adapter for real (path is fully specified in
   `src/ingest/dwg.py`'s docstring — ODA File Converter → `ezdxf` entity
   walk).
5. Delta markup: render bounding boxes for `MODIFIED`/`ADDED`/`REMOVED`
   entries back onto a copy of PID B using `pypdf` + `reportlab` overlay
   — the `Block.bbox` data needed for this already exists in the
   canonical model, so this is mostly plumbing, not new design.
6. Per-cell table bboxes (patch pdfplumber's cell geometry through
   instead of the table's outer bbox) to make table-cell citations
   pixel-precise.

## Observability, justified

A homegrown `Trace`/`Span` object (`src/observability/tracing.py`)
rather than OpenTelemetry/Langfuse/etc: this is a single-process
batch/CLI tool, not a served multi-tenant app, and a full OTel SDK +
collector is real infrastructure to stand up, run, and explain
correctly within a take-home window. The homegrown tracer captures
exactly what the assignment requires — per-stage timing, LLM
prompt/response/token/cost, one JSON file per request under `runs/` —
and is intentionally shaped (named spans with attributes, explicit
error status) so that swapping in a real OTel exporter later is a
localized change, not a rewrite.

## No secrets

`.env.example` has no real values. `GROQ_API_KEY` / `ANTHROPIC_API_KEY`
are read from the environment only (`src/chat/llm.py`); nothing is
hardcoded or logged. `git log` has no credential-bearing commits.
