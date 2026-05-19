# ──────────────────────────────────────────────
#  NMT Fine-tuning Makefile
# ──────────────────────────────────────────────

BASE_MODEL   ?= models/mbart-trimmed
ORIGIN_MODEL ?= facebook/m2m100_418M
DATASET      ?= multiun
EVAL_DATASET ?= openlanguagedata/flores_plus
PREPARED     ?= data/prepared/multiun
MAX_SAMPLES  ?= 150_000
EPOCHS       ?= 5
BATCH_SIZE   ?= 16
GRAD_ACCUM   ?= 8
LR           ?= 2e-5
MAX_SAMPLES_EVAL ?= 1500
TRIM_SAMPLES ?= 1_000_000
N_BOOT       ?= 300
GPU          ?= 0,1,2,3

# Auto-detect latest fine-tuned model (most recent models/m2m100_finetuned_* dir)
FINETUNED    ?= $(shell ls -dt models/m2m100_finetuned_* 2>/dev/null | head -1)

# ── Error analysis ─────────────────────────────
SEGMENTS_DIR ?= segments
ERROR_MODEL  ?= $(FINETUNED)
SEGMENTS_A   ?= $(shell ls -t $(SEGMENTS_DIR)/*.json 2>/dev/null | head -1)
SEGMENTS_B   ?=
TOP_N        ?= 10

# ──────────────────────────────────────────────

.PHONY: trim prepare train eval eval-base compare full smoke clean check ci ci-base \
        error-run error-run-base error-analyze error-compare error error-base

trim:
	python trim_tokenizer.py \
		--model $(ORIGIN_MODEL) \
		--output $(BASE_MODEL) \
		--sample_size $(TRIM_SAMPLES) \
		--dataset $(DATASET) 2>&1 | tee trim.log

## Prepare dataset
prepare:
	python prepare_dataset.py \
		--dataset $(DATASET) \
		--max_samples $(MAX_SAMPLES) \
		--out_dir $(PREPARED) 2>&1 | tee preparation.log

## Fine-tune
train:
	TORCH_DISTRIBUTED_DEBUG=DETAIL \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	accelerate launch --config_file cluster_config.yaml \
		train_finetune.py \
		--base_model $(BASE_MODEL) \
		--prepared_data $(PREPARED)/train \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--grad_accum_steps $(GRAD_ACCUM) \
		--lr $(LR) 2>&1 | tee fine_tuning.log

## Evaluate latest fine-tuned model (no CI)
eval:
	@echo "🔍 Evaluating: $(FINETUNED)  [GPU=$(GPU)]"
	CUDA_VISIBLE_DEVICES=$(GPU) python test.py \
		--model_path $(FINETUNED) \
		--dataset $(PREPARED)/test \
		--max_samples $(MAX_SAMPLES_EVAL) \
		--batch_size $(BATCH_SIZE) \
		--no_ci \
		--no_comet 2>&1 | tee test.log

## Evaluate base trimmed model (no CI)
eval-base:
	@echo "🔍 Evaluating base: $(BASE_MODEL)  [GPU=$(GPU)]"
	CUDA_VISIBLE_DEVICES=$(GPU) python test.py \
		--model_path $(BASE_MODEL) \
		--max_samples $(MAX_SAMPLES_EVAL) \
		--dataset $(PREPARED)/test \
		--batch_size $(BATCH_SIZE) \
		--no_ci \
		--no_comet 2>&1 | tee test_base.log

## Bootstrap CI only — best fine-tuned model (uses saved hypotheses.txt / references.txt)
## Usage:
##   make ci                                    # auto-detect latest finetuned
##   make ci FINETUNED=models/m2m100_multiun    # specific model
##   make ci N_BOOT=500                         # more iterations
ci:
	@echo "📊 Bootstrap CI: $(FINETUNED)  [GPU=$(GPU), n_boot=$(N_BOOT)]"
	CUDA_VISIBLE_DEVICES=$(GPU) python compute_ci.py \
		--model $(FINETUNED) \
		--dataset $(DATASET) \
		--n_boot $(N_BOOT) \
		--n_test $(MAX_SAMPLES_EVAL) \
		--batch_size $(BATCH_SIZE) 2>&1 | tee ci.log

## Bootstrap CI only — base trimmed model
ci-base:
	@echo "📊 Bootstrap CI: $(BASE_MODEL)  [GPU=$(GPU), n_boot=$(N_BOOT)]"
	CUDA_VISIBLE_DEVICES=$(GPU) python compute_ci.py \
		--model $(BASE_MODEL) \
		--dataset $(DATASET) \
		--n_boot $(N_BOOT) \
		--n_test $(MAX_SAMPLES_EVAL) \
		--batch_size $(BATCH_SIZE) 2>&1 | tee ci_base.log

## Run both and compare
compare: eval-base eval

## Prepare → Train → Eval fine-tuned in one shot
full: prepare train eval

check:
	@echo "🔍 Check model health: $(FINETUNED)"
	python check-save.py \
    --model_path $(FINETUNED) \
    --ref_model_path $(BASE_MODEL) 2>&1 | tee check_save.log

## Quick smoke test
smoke:
	$(MAKE) prepare MAX_SAMPLES=100
	$(MAKE) train EPOCHS=1
	$(MAKE) eval

clean:
	rm -rf $(PREPARED)
	rm -f fine_tuning.log

# ──────────────────────────────────────────────
#  Error Analysis
# ──────────────────────────────────────────────
#
#  Typical workflow:
#
#    make error                            # run + analyze latest finetuned model
#    make error-base                       # run + analyze base model
#    make error-compare \                  # head-to-head (auto-picks last 2 segment files)
#         SEGMENTS_A=segments/ft.json \
#         SEGMENTS_B=segments/base.json
#
#  Overrides:
#    ERROR_MODEL   — which model to run inference with  (default: latest finetuned)
#    SEGMENTS_DIR  — where segment JSONs are saved      (default: segments/)
#    SEGMENTS_A    — explicit segment file for analysis (default: most recent in SEGMENTS_DIR)
#    SEGMENTS_B    — second segment file for comparison
#    TOP_N         — number of worst/best segments to show in report (default: 10)
#    GPU           — visible CUDA devices               (default: 0,1,2,3)
# ──────────────────────────────────────────────

## Run inference on the latest fine-tuned model and save per-segment results
## Usage: make error-run
##        make error-run ERROR_MODEL=models/m2m100_finetuned_20260426_multiun
error-run:
	@echo "🔬 Error analysis — run: $(ERROR_MODEL)  [GPU=$(GPU)]"
	@mkdir -p $(SEGMENTS_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python error_analysis.py --run \
		--model   $(ERROR_MODEL) \
		--dataset $(PREPARED)/test \
		--n_samples $(MAX_SAMPLES_EVAL) \
		--output  $(SEGMENTS_DIR)/$(shell basename $(ERROR_MODEL))_$(shell date +%Y%m%d_%H%M%S).json \
		2>&1 | tee error_analysis.log

## Run inference on the base trimmed model and save per-segment results
## Usage: make error-run-base
error-run-base:
	@echo "🔬 Error analysis — run base: $(BASE_MODEL)  [GPU=$(GPU)]"
	@mkdir -p $(SEGMENTS_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python error_analysis.py --run \
		--model   $(BASE_MODEL) \
		--dataset $(PREPARED)/test \
		--n_samples $(MAX_SAMPLES_EVAL) \
		--output  $(SEGMENTS_DIR)/$(shell basename $(BASE_MODEL))_$(shell date +%Y%m%d_%H%M%S).json \
		2>&1 | tee error_analysis_base.log

## Analyze the most recent segment file (or an explicit one)
## Usage: make error-analyze
##        make error-analyze SEGMENTS_A=segments/opus-mt_20260515.json TOP_N=20
error-analyze:
	@echo "📋 Error analysis — report: $(SEGMENTS_A)"
	python error_analysis.py --analyze \
		--input $(SEGMENTS_A) \
		--top_n $(TOP_N)

## Compare two segment files head-to-head
## Usage: make error-compare SEGMENTS_A=segments/ft.json SEGMENTS_B=segments/base.json
error-compare:
	@if [ -z "$(SEGMENTS_B)" ]; then \
		echo "❌  SEGMENTS_B is not set. Usage: make error-compare SEGMENTS_A=... SEGMENTS_B=..."; \
		exit 1; \
	fi
	@echo "📊 Error analysis — compare:"
	@echo "   A: $(SEGMENTS_A)"
	@echo "   B: $(SEGMENTS_B)"
	python error_analysis.py --analyze \
		--input   $(SEGMENTS_A) \
		--compare $(SEGMENTS_B) \
		--top_n   $(TOP_N)

## Run inference on latest finetuned model then immediately analyze it
## Usage: make error
##        make error ERROR_MODEL=models/m2m100_finetuned_20260426_multiun
error:
	@echo "🔬 Error analysis — run + analyze: $(ERROR_MODEL)  [GPU=$(GPU)]"
	@mkdir -p $(SEGMENTS_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python error_analysis.py --run --analyze \
		--model   $(ERROR_MODEL) \
		--dataset $(PREPARED)/test \
		--n_samples $(MAX_SAMPLES_EVAL) \
		--output  $(SEGMENTS_DIR)/$(shell basename $(ERROR_MODEL))_$(shell date +%Y%m%d_%H%M%S).json \
		--top_n   $(TOP_N) \
		2>&1 | tee error_analysis.log

## Run inference on base model then immediately analyze it
## Usage: make error-base
error-base:
	@echo "🔬 Error analysis — run + analyze base: $(BASE_MODEL)  [GPU=$(GPU)]"
	@mkdir -p $(SEGMENTS_DIR)
	CUDA_VISIBLE_DEVICES=$(GPU) python error_analysis.py --run --analyze \
		--model   $(BASE_MODEL) \
		--dataset $(PREPARED)/test \
		--n_samples $(MAX_SAMPLES_EVAL) \
		--output  $(SEGMENTS_DIR)/$(shell basename $(BASE_MODEL))_$(shell date +%Y%m%d_%H%M%S).json \
		--top_n   $(TOP_N) \
		2>&1 | tee error_analysis_base.log