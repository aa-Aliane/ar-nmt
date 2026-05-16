"""
test.py
────────
Deep evaluation for AR↔EN NMT models.

Metrics
───────
  - BLEU      (sacrebleu, tokenize=flores for Arabic)
  - chrF++    (sacrebleu, word_order=2)
  - TER       (sacrebleu)
  - COMET     (Unbabel/wmt22-comet-da, requires GPU)
  - 95% CI    (bootstrap resampling, 1000 iterations by default)

Results are saved to results/<model_name>_<dataset>_<timestamp>.json
"""

import argparse
import json
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm

from src.model_loader import load_nmt_model

# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")
COMET_MODEL = "Unbabel/wmt22-comet-da"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep evaluation of a fine-tuned NMT model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_path", required=True, help="Path to the model directory.")
    parser.add_argument("--dataset",    required=True, help="Dataset name or local path.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Cap samples (-1 = all).")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument(
        "--no_comet",
        action="store_true",
        help="Skip COMET (faster, no internet required).",
    )
    parser.add_argument(
        "--src_lang", default="en",
        help="Source language code for the model (default: en).",
    )
    parser.add_argument(
        "--tgt_lang", default="ar",
        help="Target language code for the model (default: ar).",
    )
    parser.add_argument(
        "--n_boot", type=int, default=1000,
        help="Bootstrap iterations for 95%% CI (0 = skip CI).",
    )
    parser.add_argument(
        "--no_ci",
        action="store_true",
        help="Skip bootstrap confidence intervals.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_eval_dataset(dataset_name: str) -> list:
    """Load and normalise en-ar samples from supported datasets."""

    # ── Local disk path first ─────────────────────────────────────────────────
    if Path(dataset_name).exists():
        import datasets as hf_datasets
        ds_raw = hf_datasets.load_from_disk(dataset_name)
        ds = [{"translation": row["translation"]} for row in ds_raw]

    elif "flores" in dataset_name:
        ds_ar = load_dataset(dataset_name, "arb_Arab", split="devtest", streaming=True)
        ds_en = load_dataset(dataset_name, "eng_Latn", split="devtest", streaming=True)
        ds = [
            {"translation": {"ar": ar_row["text"], "en": en_row["text"]}}
            for ar_row, en_row in zip(ds_ar, ds_en)
        ]

    elif "multiun" in dataset_name:
        ds_raw = load_dataset(dataset_name, "ar-en", split="test", streaming=True)
        ds = [
            {
                "translation": {
                    "en": row["translation"]["en"],
                    "ar": row["translation"]["ar"],
                }
            }
            for row in ds_raw
        ]

    else:
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. "
            "Supported: local path, flores, multiun."
        )

    print(f"  Loaded {len(ds):,} samples from '{dataset_name}'.")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────


def translate_dataset(model, tokenizer, dataset, batch_size, lang_id):
    """Translate all English sentences and return (predictions, references, sources)."""
    predictions, references, sources = [], [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Translating"):
        batch     = dataset[i : i + batch_size]
        src_texts = [pair["translation"]["en"] for pair in batch]
        tgt_texts = [[pair["translation"]["ar"]] for pair in batch]

        inputs = tokenizer(
            src_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=lang_id,
                max_new_tokens=128,
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(tgt_texts)
        sources.extend(src_texts)

    return predictions, references, sources


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_bleu(predictions, references) -> dict:
    """SacreBLEU with flores tokeniser (recommended for Arabic)."""
    metric = evaluate.load("sacrebleu")
    result = metric.compute(
        predictions=predictions,
        references=references,
        tokenize="flores101",
    )
    return {
        "bleu":       round(result["score"], 4),
        "bleu_1":     round(result["precisions"][0], 4),
        "bleu_2":     round(result["precisions"][1], 4),
        "bleu_3":     round(result["precisions"][2], 4),
        "bleu_4":     round(result["precisions"][3], 4),
        "brevity_penalty": round(result["bp"], 4),
    }


def compute_chrf(predictions, references) -> dict:
    """chrF++ (word_order=2) — better than BLEU for morphologically rich languages."""
    metric = evaluate.load("chrf")
    result = metric.compute(
        predictions=predictions,
        references=references,
        word_order=2,
        char_order=6,
    )
    return {"chrF++": round(result["score"], 4)}


def compute_ter(predictions, references) -> dict:
    """Translation Edit Rate — lower is better."""
    metric = evaluate.load("ter")
    result = metric.compute(
        predictions=predictions,
        references=references,
        normalized=True,
        support_zh_ja_chars=False,
    )
    return {"ter": round(result["score"], 4)}


def compute_comet(sources, predictions, references) -> dict:
    """
    COMET (wmt22-comet-da) — neural metric, best correlation with human judgements.
    Requires ~1.7 GB download on first run.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("  [!] COMET not installed. Run: pip install unbabel-comet")
        return {"comet": None, "comet_error": "unbabel-comet not installed"}

    print(f"  Loading COMET model ({COMET_MODEL}) …")
    model_path = download_model(COMET_MODEL)
    comet_model = load_from_checkpoint(model_path)

    flat_refs = [r[0] if isinstance(r, list) else r for r in references]
    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, flat_refs)
    ]

    output = comet_model.predict(data, batch_size=16, gpus=1 if DEVICE == "cuda" else 0)
    mean_score = round(float(output.system_score), 4)
    return {"comet": mean_score}


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────


def _boot_sample(hypotheses, references, idx):
    """Return a resampled (hyp, ref) pair given bootstrap indices."""
    h = [hypotheses[i] for i in idx]
    # references may be list-of-lists (sacrebleu format) or flat strings
    r = [references[i] for i in idx]
    return h, r


def compute_bootstrap_ci(
    hypotheses: list,
    references: list,       # list of [str] (sacrebleu format)
    sources:    list,
    n_boot:     int = 1000,
    alpha:      float = 0.05,
    run_comet:  bool = False,
) -> dict:
    """
    Bootstrap resampling (n_boot iterations) to estimate 95% CI for
    BLEU, chrF++, TER, and optionally COMET.

    Returns a dict shaped like:
      {
        "bleu":   {"mean": x, "ci_low": y, "ci_high": z},
        "chrf":   {...},
        "ter":    {...},
        "comet":  {...},   # only if run_comet=True
      }
    """
    n = len(hypotheses)
    # flat refs for sacrebleu bootstrap (list-of-lists → list)
    flat_refs = [r[0] if isinstance(r, list) else r for r in references]

    bleu_m = BLEU(effective_order=True, tokenize="flores101")
    chrf_m = CHRF(word_order=2)
    ter_m  = TER()

    bleu_scores, chrf_scores, ter_scores = [], [], []

    print(f"\n  Bootstrap CI ({n_boot} iterations) …")
    for _ in tqdm(range(n_boot), desc="Bootstrapping"):
        idx = np.random.choice(n, n, replace=True)
        h = [hypotheses[i] for i in idx]
        r = [flat_refs[i]  for i in idx]

        bleu_scores.append(bleu_m.corpus_score(h, [r]).score)
        chrf_scores.append(chrf_m.corpus_score(h, [r]).score)
        ter_scores.append(ter_m.corpus_score(h,  [r]).score)

    def _ci(scores):
        return {
            "mean":    round(float(np.mean(scores)), 4),
            "ci_low":  round(float(np.percentile(scores, 100 * alpha / 2)), 4),
            "ci_high": round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4),
        }

    ci = {
        "bleu": _ci(bleu_scores),
        "chrf": _ci(chrf_scores),
        "ter":  _ci(ter_scores),
    }

    if run_comet:
        try:
            from comet import download_model, load_from_checkpoint
            print(f"  Loading COMET for bootstrap …")
            comet_model = load_from_checkpoint(download_model(COMET_MODEL))
            comet_scores = []

            for _ in tqdm(range(n_boot), desc="COMET bootstrap"):
                idx  = np.random.choice(n, n, replace=True)
                data = [
                    {"src": sources[i], "mt": hypotheses[i], "ref": flat_refs[i]}
                    for i in idx
                ]
                result = comet_model.predict(
                    data, batch_size=16, gpus=1 if DEVICE == "cuda" else 0
                )
                comet_scores.append(result.system_score)

            ci["comet"] = _ci(comet_scores)

        except ImportError:
            print("  [!] COMET not installed — skipping COMET CI.")
            ci["comet"] = None

    return ci


# ─────────────────────────────────────────────────────────────────────────────
# Result persistence
# ─────────────────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    """Turn an arbitrary path/name into a filesystem-safe slug."""
    text = Path(text).name if "/" in text or "\\" in text else text
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")


def save_results(
    model_path: str,
    dataset_name: str,
    scores: dict,
    n_samples: int,
    confidence_intervals: dict | None = None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _slug(model_path)
    data_slug  = _slug(dataset_name)
    out_path   = RESULTS_DIR / f"{model_slug}__{data_slug}__{timestamp}.json"

    payload = {
        "model_path":   str(model_path),
        "dataset":      str(dataset_name),
        "n_samples":    n_samples,
        "timestamp":    timestamp,
        "hostname":     socket.gethostname(),
        "device":       DEVICE,
        "scores":       scores,
        "confidence_intervals": confidence_intervals,  # None if --no_ci
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = load_eval_dataset(args.dataset)
    if args.max_samples > 0:
        ds = ds[: args.max_samples]
    print(f"  Evaluating on {len(ds):,} samples.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n  Loading model from '{args.model_path}' …")
    model, tokenizer = load_nmt_model(args.model_path, device=DEVICE)
    model.eval()

    is_nllb  = "facebook/nllb" in args.model_path.lower() or "facebook/nllb" in type(tokenizer).__name__.lower()
    is_mbart = "mbart" in args.model_path.lower() or "mbart" in type(tokenizer).__name__.lower()

    NLLB_CODES  = {"en": "eng_Latn", "ar": "arb_Arab"}
    MBART_CODES = {"en": "en_XX",    "ar": "ar_AR"}

    if is_nllb:
        src_lang = NLLB_CODES.get(args.src_lang, args.src_lang)
        tgt_lang = NLLB_CODES.get(args.tgt_lang, args.tgt_lang)
    elif is_mbart:
        src_lang = MBART_CODES.get(args.src_lang, args.src_lang)
        tgt_lang = MBART_CODES.get(args.tgt_lang, args.tgt_lang)
    else:
        src_lang = args.src_lang
        tgt_lang = args.tgt_lang

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    if hasattr(tokenizer, "get_lang_id"):
        lang_id = tokenizer.get_lang_id(tgt_lang)
    else:
        lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # ── Translate ─────────────────────────────────────────────────────────────
    predictions, references, sources = translate_dataset(
        model, tokenizer, ds, args.batch_size, lang_id
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    scores = {}

    print("\n  Computing BLEU …")
    scores.update(compute_bleu(predictions, references))

    print("  Computing chrF++ …")
    scores.update(compute_chrf(predictions, references))

    print("  Computing TER …")
    scores.update(compute_ter(predictions, references))

    if not args.no_comet:
        print("  Computing COMET …")
        scores.update(compute_comet(sources, predictions, references))
    else:
        print("  Skipping COMET (--no_comet).")

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    confidence_intervals = None
    if not args.no_ci and args.n_boot > 0:
        confidence_intervals = compute_bootstrap_ci(
            hypotheses=predictions,
            references=references,
            sources=sources,
            n_boot=args.n_boot,
            run_comet=not args.no_comet,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    for k, v in scores.items():
        print(f"  {k:<20} {v}")

    if confidence_intervals:
        print("\n  95% CONFIDENCE INTERVALS  (bootstrap, n={})".format(args.n_boot))
        print("  {:<8} {:>8}  {:>8}  {:>8}".format("Metric", "Mean", "CI low", "CI high"))
        print("  " + "-" * 38)
        labels = {"bleu": "BLEU ↑", "chrf": "chrF++ ↑", "ter": "TER ↓", "comet": "COMET ↑"}
        for key, label in labels.items():
            if key in confidence_intervals and confidence_intervals[key]:
                c = confidence_intervals[key]
                print(f"  {label:<10} {c['mean']:>8.4f}  {c['ci_low']:>8.4f}  {c['ci_high']:>8.4f}")

    print("=" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = save_results(
        args.model_path,
        args.dataset,
        scores,
        len(ds),
    )
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()