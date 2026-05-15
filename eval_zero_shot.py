import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_loader import get_dataset
from src.metrics import compute_metrics
from src.model_loader import load_nmt_model

# ──────────────────────────────────────────────────────────────
# Model list
# ──────────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "facebook/m2m100_418M",  # 0
    "facebook/mbart-large-50-many-to-many-mmt",  # 1
    "facebook/nllb-200-distilled-600M",  # 2
    "Helsinki-NLP/opus-mt-en-ar",  # 3
]

# ──────────────────────────────────────────────────────────────
# Per-model overrides
# Large / high-vocab models need reduced batch & beams to avoid OOM.
# Models not listed here fall back to the defaults below.
# ──────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "facebook/mbart-large-50-many-to-many-mmt": {
        "batch_size": 8,
        "num_beams": 2,
    },
    "facebook/nllb-200-distilled-600M": {
        "batch_size": 4,
        "num_beams": 2,
    },
}

# Defaults applied to all models not in MODEL_CONFIGS
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
RESULTS_DIR = Path("results")

DATASETS_TO_TEST = ["opus-100", "multiun"]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot NMT evaluation")
    parser.add_argument(
        "--model",
        type=int,
        default=None,
        metavar="INDEX",
        help=(
            "Index of the model to evaluate (0-based). "
            "If omitted, all models are evaluated.\n"
            + "\n".join(f"  {i}: {m}" for i, m in enumerate(MODELS_TO_TEST))
        ),
    )
    return parser.parse_args()


def get_forced_bos_token_id(model_name: str, tokenizer) -> int | None:
    """
    Return the target-language BOS token id required by each architecture.
    OpusMT handles language forcing internally — returns None.
    """
    name_lower = model_name.lower()
    if "m2m100" in name_lower:
        return tokenizer.get_lang_id("ar")
    if "mbart" in name_lower:
        return tokenizer.lang_code_to_id["ar_AR"]
    if "nllb" in name_lower:
        # lang_code_to_id does not exist on NllbTokenizer — use convert_tokens_to_ids
        return tokenizer.convert_tokens_to_ids("arb_Arab")
    return None  # OpusMT


def collate_fn(batch):
    """Extract En→Ar pairs from HF translation datasets."""
    sources = [ex["translation"]["en"] for ex in batch]
    targets = [ex["translation"]["ar"] for ex in batch]
    return sources, targets


def log_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def free_vram(model, accelerator):
    """Aggressively release GPU memory between models."""
    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    accelerator.free_memory()


# ──────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────
def run_zero_shot(models: list[str]):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    is_main = accelerator.is_main_process

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for model_name in models:

        # ── Per-model batch / beam config ──────────────────────────
        cfg = MODEL_CONFIGS.get(model_name, {})
        batch_size = cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        num_beams = cfg.get("num_beams", DEFAULT_NUM_BEAMS)

        if is_main:
            print(f"\n{'='*62}")
            print(f"  Model      : {model_name}")
            print(f"  Batch/GPU  : {batch_size}   |   Beams: {num_beams}")
            print(f"{'='*62}")

        # ── Load model ─────────────────────────────────────────────
        # NOTE: model is NOT passed to accelerator.prepare() — doing so wraps
        # it in DDP which allocates a full gradient buffer (~= model size) even
        # during inference, causing OOM on 10-12 GB GPUs. For inference we only
        # need each rank to hold the model on its own device; the dataloader
        # prepare() handles data sharding and gather_object() collects results.
        model, tokenizer = load_nmt_model(model_name, device)
        forced_bos = get_forced_bos_token_id(model_name, tokenizer)
        model.eval()

        model_results = {}

        for ds_name in DATASETS_TO_TEST:
            if is_main:
                print(f"\n  Dataset: {ds_name}")

            # ── Load dataset ───────────────────────────────────────
            try:
                ds = get_dataset(ds_name)
            except Exception as e:
                if is_main:
                    print(f"  [SKIP] Could not load '{ds_name}': {e}")
                continue

            dataloader = DataLoader(
                ds,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
            )
            # Only prepare the dataloader — not the model
            dataloader = accelerator.prepare(dataloader)

            all_preds: list[str] = []
            all_labels: list[str] = []

            # ── Inference ──────────────────────────────────────────
            for sources, targets in tqdm(
                dataloader,
                desc=f"[{model_name.split('/')[-1]}] {ds_name}",
                disable=not is_main,
            ):
                inputs = tokenizer(
                    sources,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                generate_kwargs = dict(
                    **inputs,
                    num_beams=num_beams,
                    max_new_tokens=MAX_NEW_TOKENS,
                    early_stopping=True,
                )
                if forced_bos is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos

                with torch.no_grad():
                    # Model is not DDP-wrapped, call .generate() directly
                    generated_ids = model.generate(**generate_kwargs)

                decoded = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                gathered_preds = gather_object(decoded)
                gathered_labels = gather_object(list(targets))

                if is_main:
                    all_preds.extend(gathered_preds)
                    all_labels.extend(gathered_labels)

            # ── Metrics (main process only) ────────────────────────
            if is_main:
                if not all_preds:
                    print("  [WARN] No predictions collected, skipping metrics.")
                    continue

                metrics = compute_metrics(all_preds, all_labels)
                model_results[ds_name] = metrics

                print(f"\n  Results — {model_name} on {ds_name}:")
                print(f"    SacreBLEU    : {metrics['bleu']:.2f}")
                print(f"    TER          : {metrics['ter']:.2f}")
                print(f"    BERTScore-F1 : {metrics['bert_score_f1']:.4f}")
                print(f"    Length Ratio : {metrics['length_ratio']:.3f}")

        if is_main:
            all_results[model_name] = model_results

        # ── Free VRAM before next model ────────────────────────────
        free_vram(model, accelerator)

    # ── Save JSON results ──────────────────────────────────────────
    if is_main:
        out_path = RESULTS_DIR / f"zero_shot_{run_id}.json"
        log_results(all_results, out_path)
        print(f"\n✓ Results saved to {out_path}")

        # ── Summary table ──────────────────────────────────────────
        col_m = 42
        col_d = 10
        header = (
            f"  {'Model':<{col_m}} {'Dataset':<{col_d}}"
            f" {'BLEU':>6}  {'TER':>6}  {'BERT-F1':>8}  {'LenR':>5}"
        )
        sep = f"  {'-'*col_m} {'-'*col_d} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}"

        print(f"\n{'='*62}")
        print("  ZERO-SHOT EVALUATION SUMMARY")
        print(f"{'='*62}")
        print(header)
        print(sep)

        for m, ds_dict in all_results.items():
            short = m.split("/")[-1]
            for ds, met in ds_dict.items():
                print(
                    f"  {short:<{col_m}} {ds:<{col_d}}"
                    f" {met['bleu']:>6.2f}"
                    f"  {met['ter']:>6.2f}"
                    f"  {met['bert_score_f1']:>8.4f}"
                    f"  {met['length_ratio']:>5.3f}"
                )


if __name__ == "__main__":
    args = parse_args()

    if args.model is not None:
        if args.model < 0 or args.model >= len(MODELS_TO_TEST):
            raise SystemExit(
                f"Error: --model index {args.model} is out of range. "
                f"Valid range: 0–{len(MODELS_TO_TEST) - 1}."
            )
        models_to_run = [MODELS_TO_TEST[args.model]]
    else:
        models_to_run = MODELS_TO_TEST

    run_zero_shot(models_to_run)
