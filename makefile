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


# Auto-detect latest fine-tuned model (most recent models/m2m100_finetuned_* dir)
FINETUNED    ?= $(shell ls -dt models/m2m100_finetuned_* 2>/dev/null | head -1)

# ──────────────────────────────────────────────

.PHONY: trim prepare train eval eval-base compare full smoke clean

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

## Evaluate latest fine-tuned model
eval:
	@echo "🔍 Evaluating: $(FINETUNED)"
	python test.py \
		--model_path $(FINETUNED) \
		--dataset $(PREPARED)/test \
		--max_samples $(MAX_SAMPLES_EVAL) \
		--batch_size $(BATCH_SIZE) 2>&1 | tee test.log

## Evaluate base trimmed model
eval-base:
	@echo "🔍 Evaluating base: $(BASE_MODEL)"
	python test.py \
		--model_path $(BASE_MODEL) \
		--max_samples $(MAX_SAMPLES_EVAL) \
		--dataset $(PREPARED)/test \
		--batch_size $(BATCH_SIZE) 2>&1 | tee test.log


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
