.PHONY: install samples run chat eval test clean markup

PY := python3

install:
	pip install -r requirements.txt --break-system-packages

# Regenerate the synthetic sample document pairs + their ground truth.
# Already committed under data/ and eval/datasets/, so this is only
# needed if you edit scripts/generate_samples.py.
samples:
	$(PY) scripts/generate_samples.py

# Reproducible single-command run: ingest a pair, produce a delta report.
# Defaults to data/samples/pair_A_equipment_schedule if no args given.
run:
	$(PY) scripts/run_pipeline.py --out out/pair_A

run-pair-b:
	$(PY) scripts/run_pipeline.py \
		--pid-a 26-KA-902-RevA --path-a data/samples/pair_B_valve_notes/26-KA-902_RevA.pdf \
		--pid-b 26-KA-902-RevB --path-b data/samples/pair_B_valve_notes/26-KA-902_RevB.pdf \
		--out out/pair_B

run-pair-c-scanned:
	$(PY) scripts/run_pipeline.py \
		--pid-a 26-KA-901-RevA-scanned --path-a data/samples/pair_C_cross_document/26-KA-901_RevA_SCANNED.pdf \
		--pid-b 26-KA-901-RevB-native --path-b data/samples/pair_C_cross_document/26-KA-901_RevB.pdf \
		--out out/pair_C

# Interactive grounded chat over pair_A. Use `make chat Q="..."` for
# single-shot (non-interactive) mode.
chat:
ifdef Q
	$(PY) scripts/chat.py -q "$(Q)"
else
	$(PY) scripts/chat.py
endif

# Runs the full eval harness and prints a scorecard.
eval:
	$(PY) eval/run_eval.py

test:
	$(PY) -m pytest tests/ -v

markup:
	@echo "Delta markup (bonus) was cut for this submission -- see README 'what we cut'."

clean:
	rm -rf out runs eval/results __pycache__ src/**/__pycache__ .pytest_cache
