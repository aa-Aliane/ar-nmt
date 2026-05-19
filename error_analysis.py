"""
error_analysis.py
─────────────────
Two-mode error analysis tool for EN→AR NMT experiments.

MODE 1 — --run
  Re-runs inference on a dataset and saves per-segment results to a JSON.
  This is the one step that costs GPU time (~2-5 min per model for 1500 samples).
  Run it once per model; subsequent analysis is instant.

MODE 2 — --analyze
  Loads one or two segment JSON files and produces a full error report:
    • Score distribution (good / medium / poor)
    • Length bucket analysis (short / medium / long source)
    • Worst N segments
    • Empty / repetitive output detection (Arabic-specific)
    • Head-to-head comparison when two models are provided

Usage
─────
  # Step 1 — generate segment data (once per model)
  python error_analysis.py --run \\
      --model Helsinki-NLP/opus-mt-en-ar \\
      --dataset data/prepared/multiun/test \\
      --n_samples 1500 \\
      --output segments/opus-mt.json

  # Step 2a — analyze a single model
  python error_analysis.py --analyze --input segments/opus-mt.json

  # Step 2b — compare two models
  python error_analysis.py --analyze \\
      --input segments/opus-mt.json \\
      --compare segments/m2m100_finetuned.json

  # Both steps in one command
  python error_analysis.py --run --analyze \\
      --model Helsinki-NLP/opus-mt-en-ar \\
      --dataset data/prepared/multiun/test \\
      --output segments/opus-mt.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHRF_METRIC = CHRF(word_order=2)  # chrF++

# Score thresholds for bucketing
GOOD_THRESHOLD   = 50.0   # chrF++ ≥ 50 → good
MEDIUM_THRESHOLD = 25.0   # chrF++ 25-50 → medium  |  < 25 → poor

# Source length thresholds (in whitespace-split tokens)
SHORT_MAX  = 15
LONG_MIN   = 35

# Repetition: flag if any single token repeats > this many times in a row
REPETITION_MAX = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — supports all model families in this project
# ─────────────────────────────────────────────────────────────────────────────

def _detect_family(model_path: str) -> str:
    p = model_path.lower()
    if "opus-mt" in p or "helsinki" in p:
        return "marian"
    if "nllb" in p:
        return "nllb"
    if "mbart" in p:
        return "mbart"
    return "m2m100"   # covers trimmed and fine-tuned variants


def load_model_and_tokenizer(model_path: str):
    """
    Return (model, tokenizer, family) for any model in this project.
    Handles RemappedTokenizer automatically for trimmed M2M-100 variants.
    """
    family = _detect_family(model_path)

    if family == "marian":
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model     = MarianMTModel.from_pretrained(model_path).to(DEVICE)
        return model, tokenizer, family

    if family == "nllb":
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
        tokenizer = NllbTokenizerFast.from_pretrained(
            model_path, src_lang="eng_Latn", tgt_lang="arb_Arab"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
        return model, tokenizer, family

    if family == "mbart":
        from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
        # Pass src_lang/tgt_lang directly to from_pretrained so they override
        # whatever is stored in the saved tokenizer_config.json (which may
        # have a bare "en" instead of the required "en_XX" locale code).
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_path, src_lang="en_XX", tgt_lang="ar_AR"
        )
        model = MBartForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
        return model, tokenizer, family

    # M2M-100 (original, trimmed, or fine-tuned) ──────────────────────────────
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    # Try to use the project's src utilities for RemappedTokenizer support
    try:
        from src.model_loader import load_nmt_model
        model, tokenizer = load_nmt_model(model_path, device=DEVICE)
        return model, tokenizer, family
    except Exception:
        pass

    # Fallback: plain M2M100 loader (no trimmed-vocab support)
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "ar"
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    return model, tokenizer, family


def get_forced_bos(tokenizer, family: str) -> int | None:
    """Return the forced_bos_token_id for Arabic, family-aware."""
    if family == "marian":
        return None   # Marian sets target via tokenizer prefix, no forced_bos
    if family == "nllb":
        return tokenizer.convert_tokens_to_ids("arb_Arab")
    if family == "mbart":
        return tokenizer.convert_tokens_to_ids("ar_AR")
    # M2M-100 (plain or RemappedTokenizer)
    try:
        return tokenizer.get_lang_id("ar")
    except Exception:
        return tokenizer.convert_tokens_to_ids("__ar__")


def translate_batch(model, tokenizer, family: str, src_text: str) -> str:
    """Translate a single source sentence and return the decoded string."""
    if family == "marian":
        inputs = tokenizer(
            [src_text], return_tensors="pt", truncation=True, padding=True
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    # Seq2seq families that need forced_bos ───────────────────────────────────
    forced_bos = get_forced_bos(tokenizer, family)

    # Check for RemappedTokenizer (trimmed M2M-100 vocab)
    try:
        from src.tokenizer_utils import RemappedTokenizer
        if isinstance(tokenizer, RemappedTokenizer):
            base_tok   = object.__getattribute__(tokenizer, "_tokenizer")
            fwd_tensor = object.__getattribute__(tokenizer, "_fwd_tensor").to(DEVICE)
            inputs = base_tok(
                src_text, return_tensors="pt", truncation=True, padding=True
            ).to(DEVICE)
            inputs["input_ids"] = fwd_tensor[
                inputs["input_ids"].clamp(0, len(fwd_tensor) - 1)
            ]
            with torch.no_grad():
                out = model.generate(
                    **inputs, forced_bos_token_id=forced_bos,
                    max_new_tokens=128, num_beams=4,
                )
            return tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    except ImportError:
        pass

    # Standard seq2seq path ───────────────────────────────────────────────────
    inputs = tokenizer(
        src_text, return_tensors="pt", truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, forced_bos_token_id=forced_bos,
            max_new_tokens=128, num_beams=4,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading — local (save_to_disk) or HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_samples(dataset_path: str, n_samples: int) -> list[dict]:
    """
    Return a list of {"en": ..., "ar": ...} dicts.
    Accepts:
      • A local path produced by dataset.save_to_disk()  (Arrow format)
      • A HuggingFace Hub identifier like "Helsinki-NLP/opus-100"
      • A local .jsonl file with {"translation": {"en": ..., "ar": ...}} lines
    """
    path = Path(dataset_path)

    # ── Local Arrow dataset ───────────────────────────────────────────────────
    if path.exists() and (path / "dataset_info.json").exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(path))
        samples = []
        for i in range(min(n_samples, len(ds))):
            row = ds[i]
            # Support both flat {"en":..,"ar":..} and nested {"translation":{..}}
            if "translation" in row:
                samples.append(row["translation"])
            else:
                samples.append({"en": row["en"], "ar": row["ar"]})
        return samples

    # ── Local JSONL ───────────────────────────────────────────────────────────
    if path.exists() and path.suffix in {".jsonl", ".json"}:
        samples = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                t = row.get("translation", row)
                samples.append({"en": t["en"], "ar": t["ar"]})
                if len(samples) >= n_samples:
                    break
        return samples

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"Loading '{dataset_path}' from HuggingFace Hub …")
    # Guess the right config name
    if "multiun" in dataset_path.lower():
        ds = load_dataset(dataset_path, "ar-en", split=f"train[:{n_samples}]")
    elif "opus-100" in dataset_path.lower():
        ds = load_dataset(dataset_path, "ar-en", split=f"test[:{n_samples}]")
    else:
        ds = load_dataset(dataset_path, split=f"test[:{n_samples}]")

    return [row["translation"] for row in ds]


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment helpers
# ─────────────────────────────────────────────────────────────────────────────

def segment_chrf(hypothesis: str, reference: str) -> float:
    return CHRF_METRIC.sentence_score(hypothesis, [reference]).score


def is_empty(text: str) -> bool:
    return len(text.strip()) == 0


def is_repetitive(text: str, max_repeat: int = REPETITION_MAX) -> bool:
    """Detect token-level repetition (e.g. 'iveiveiveive …')."""
    tokens = text.split()
    if len(tokens) < max_repeat:
        return False
    for i in range(len(tokens) - max_repeat + 1):
        if len(set(tokens[i : i + max_repeat])) == 1:
            return True
    # Also catch character-level repetition for Arabic glued tokens
    if re.search(r"(.{2,8})\1{5,}", text):
        return True
    return False


def length_bucket(src_text: str) -> str:
    n = len(src_text.split())
    if n <= SHORT_MAX:
        return "short"
    if n >= LONG_MIN:
        return "long"
    return "medium"


def score_bucket(chrf: float) -> str:
    if chrf >= GOOD_THRESHOLD:
        return "good"
    if chrf >= MEDIUM_THRESHOLD:
        return "medium"
    return "poor"


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Run inference and save segments
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args) -> list[dict]:
    print(f"\n{'='*60}")
    print(f" RUN MODE — {args.model}")
    print(f"{'='*60}")

    samples = load_dataset_samples(args.dataset, args.n_samples)
    print(f" Loaded {len(samples)} samples from '{args.dataset}'")

    model, tokenizer, family = load_model_and_tokenizer(args.model)
    model.eval()
    print(f" Model family : {family}  |  device : {DEVICE}\n")

    segments = []
    errors   = 0

    for i, pair in enumerate(tqdm(samples, desc="Translating")):
        src = pair["en"]
        ref = pair["ar"]

        try:
            hyp = translate_batch(model, tokenizer, family, src)
        except Exception as exc:
            hyp = ""
            errors += 1
            if errors <= 5:
                print(f"\n  [!] Error on sample {i}: {exc}")

        chrf = segment_chrf(hyp, ref)

        segments.append({
            "id"         : i,
            "src"        : src,
            "ref"        : ref,
            "hyp"        : hyp,
            "chrf"       : round(chrf, 4),
            "src_len"    : len(src.split()),
            "hyp_len"    : len(hyp.split()),
            "ref_len"    : len(ref.split()),
            "empty"      : is_empty(hyp),
            "repetitive" : is_repetitive(hyp),
            "len_bucket" : length_bucket(src),
            "score_bucket": score_bucket(chrf),
        })

    if errors:
        print(f"\n  [!] {errors} segments failed inference and were saved as empty.")

    # Save ─────────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "model"     : args.model,
        "dataset"   : args.dataset,
        "n_samples" : len(segments),
        "timestamp" : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device"    : DEVICE,
        "family"    : family,
    }
    out = {"meta": meta, "segments": segments}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n  Segments saved → {output_path}")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Analyze saved segments
# ─────────────────────────────────────────────────────────────────────────────

def load_segments(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["meta"], data["segments"]


# ── Single-model report ───────────────────────────────────────────────────────

def report_single(meta: dict, segments: list[dict], top_n: int = 10):
    label = Path(meta["model"]).name or meta["model"]
    total = len(segments)
    chrfs = [s["chrf"] for s in segments]

    print(f"\n{'='*60}")
    print(f" REPORT — {label}")
    print(f" Dataset : {meta['dataset']}  |  n={total}  |  {meta['timestamp']}")
    print(f"{'='*60}")

    # ── Overall scores ─────────────────────────────────────────────────────────
    avg_chrf = sum(chrfs) / total
    print(f"\n── Overall ──────────────────────────────────────")
    print(f"  Avg chrF++ : {avg_chrf:.2f}")
    print(f"  Min chrF++ : {min(chrfs):.2f}")
    print(f"  Max chrF++ : {max(chrfs):.2f}")

    # ── Quality buckets ────────────────────────────────────────────────────────
    buckets = {"good": 0, "medium": 0, "poor": 0}
    for s in segments:
        buckets[s["score_bucket"]] += 1

    print(f"\n── Score Buckets (chrF++) ───────────────────────")
    print(f"  Good   (≥{GOOD_THRESHOLD})   : {buckets['good']:>5}  ({buckets['good']/total*100:.1f}%)")
    print(f"  Medium ({MEDIUM_THRESHOLD}–{GOOD_THRESHOLD}) : {buckets['medium']:>5}  ({buckets['medium']/total*100:.1f}%)")
    print(f"  Poor   (<{MEDIUM_THRESHOLD})   : {buckets['poor']:>5}  ({buckets['poor']/total*100:.1f}%)")

    # ── Length bucket analysis ─────────────────────────────────────────────────
    print(f"\n── Length Buckets (source tokens) ───────────────")
    print(f"  {'Bucket':<10}  {'Count':>6}  {'Avg chrF++':>10}  {'% Poor':>7}")
    print(f"  {'-'*40}")
    for lb in ["short", "medium", "long"]:
        segs = [s for s in segments if s["len_bucket"] == lb]
        if not segs:
            continue
        avg  = sum(s["chrf"] for s in segs) / len(segs)
        poor = sum(1 for s in segs if s["score_bucket"] == "poor")
        thresholds = f"(≤{SHORT_MAX})" if lb == "short" else f"(≥{LONG_MIN})" if lb == "long" else f"({SHORT_MAX+1}–{LONG_MIN-1})"
        print(f"  {lb:<10}  {len(segs):>6}  {avg:>10.2f}  {poor/len(segs)*100:>6.1f}%  {thresholds}")

    # ── Arabic-specific issues ─────────────────────────────────────────────────
    empty      = [s for s in segments if s["empty"]]
    repetitive = [s for s in segments if s["repetitive"]]
    print(f"\n── Arabic-Specific Issues ───────────────────────")
    print(f"  Empty outputs      : {len(empty):>5}  ({len(empty)/total*100:.1f}%)")
    print(f"  Repetitive outputs : {len(repetitive):>5}  ({len(repetitive)/total*100:.1f}%)")

    if repetitive:
        print(f"\n  Sample repetitive outputs (first 3):")
        for s in repetitive[:3]:
            print(f"    SRC : {s['src'][:80]}")
            print(f"    HYP : {s['hyp'][:120]}")
            print()

    # ── Length ratio analysis ──────────────────────────────────────────────────
    ratios = [s["hyp_len"] / max(s["ref_len"], 1) for s in segments if not s["empty"]]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    too_short = sum(1 for r in ratios if r < 0.5)
    too_long  = sum(1 for r in ratios if r > 2.0)
    print(f"\n── Output Length Ratio (hyp_len / ref_len) ─────")
    print(f"  Average ratio : {avg_ratio:.2f}  (1.0 = perfect)")
    print(f"  Too short (<0.5x ref) : {too_short}  ({too_short/total*100:.1f}%)")
    print(f"  Too long  (>2.0x ref) : {too_long}  ({too_long/total*100:.1f}%)")

    # ── Worst segments ─────────────────────────────────────────────────────────
    worst = sorted(segments, key=lambda s: s["chrf"])[:top_n]
    print(f"\n── Worst {top_n} Segments (by chrF++) ───────────────")
    for rank, s in enumerate(worst, 1):
        print(f"\n  [{rank}] chrF++={s['chrf']:.1f}  src_len={s['src_len']}  "
              f"{'EMPTY' if s['empty'] else ''}  {'REPET.' if s['repetitive'] else ''}")
        print(f"    SRC : {s['src'][:100]}")
        print(f"    REF : {s['ref'][:100]}")
        print(f"    HYP : {s['hyp'][:100]}")


# ── Two-model comparison ──────────────────────────────────────────────────────

def report_compare(
    meta_a: dict, segs_a: list[dict],
    meta_b: dict, segs_b: list[dict],
    top_n: int = 10,
):
    label_a = Path(meta_a["model"]).name or meta_a["model"]
    label_b = Path(meta_b["model"]).name or meta_b["model"]

    # Align by segment id
    map_b = {s["id"]: s for s in segs_b}
    paired = [(s, map_b[s["id"]]) for s in segs_a if s["id"] in map_b]

    if not paired:
        print("[!] No overlapping segment IDs — cannot compare. "
              "Make sure both files were run on the same dataset.")
        return

    total = len(paired)
    chrfs_a = [p[0]["chrf"] for p in paired]
    chrfs_b = [p[1]["chrf"] for p in paired]
    deltas  = [b - a for a, b in zip(chrfs_a, chrfs_b)]

    print(f"\n{'='*60}")
    print(f" COMPARISON")
    print(f"  A : {label_a}")
    print(f"  B : {label_b}")
    print(f"  n : {total} aligned segments")
    print(f"{'='*60}")

    print(f"\n── Overall ──────────────────────────────────────")
    print(f"  {'Model':<40}  {'Avg chrF++':>10}")
    print(f"  {label_a:<40}  {sum(chrfs_a)/total:>10.2f}")
    print(f"  {label_b:<40}  {sum(chrfs_b)/total:>10.2f}")
    winner = label_b if sum(chrfs_b) > sum(chrfs_a) else label_a
    print(f"  → {winner} wins overall")

    # ── Win/loss breakdown ─────────────────────────────────────────────────────
    a_wins = sum(1 for d in deltas if d < -2)   # A better by >2 chrF
    b_wins = sum(1 for d in deltas if d >  2)   # B better by >2 chrF
    ties   = total - a_wins - b_wins

    print(f"\n── Win / Tie / Loss (margin > 2 chrF++) ─────────")
    print(f"  A wins : {a_wins:>5}  ({a_wins/total*100:.1f}%)")
    print(f"  Tie    : {ties:>5}  ({ties/total*100:.1f}%)")
    print(f"  B wins : {b_wins:>5}  ({b_wins/total*100:.1f}%)")

    # ── Per length bucket ──────────────────────────────────────────────────────
    print(f"\n── Avg chrF++ by Source Length ──────────────────")
    print(f"  {'Bucket':<10}  {'A':>8}  {'B':>8}  {'Delta (B-A)':>12}")
    print(f"  {'-'*44}")
    for lb in ["short", "medium", "long"]:
        sub = [(a, b) for a, b in paired if a["len_bucket"] == lb]
        if not sub:
            continue
        avg_a = sum(p[0]["chrf"] for p in sub) / len(sub)
        avg_b = sum(p[1]["chrf"] for p in sub) / len(sub)
        print(f"  {lb:<10}  {avg_a:>8.2f}  {avg_b:>8.2f}  {avg_b - avg_a:>+12.2f}")

    # ── Segments where B is much better than A ─────────────────────────────────
    b_much_better = sorted(
        [(a, b) for a, b in paired if b["chrf"] - a["chrf"] > 10],
        key=lambda p: -(p[1]["chrf"] - p[0]["chrf"])
    )[:top_n]

    if b_much_better:
        print(f"\n── Top segments where B >> A (chrF++ gap > 10) ──")
        for a, b in b_much_better[:5]:
            print(f"\n  Δ={b['chrf']-a['chrf']:+.1f}  A={a['chrf']:.1f}  B={b['chrf']:.1f}")
            print(f"    SRC : {a['src'][:100]}")
            print(f"    REF : {a['ref'][:100]}")
            print(f"    A   : {a['hyp'][:100]}")
            print(f"    B   : {b['hyp'][:100]}")

    # ── Segments where A is much better than B ─────────────────────────────────
    a_much_better = sorted(
        [(a, b) for a, b in paired if a["chrf"] - b["chrf"] > 10],
        key=lambda p: -(p[0]["chrf"] - p[1]["chrf"])
    )[:top_n]

    if a_much_better:
        print(f"\n── Top segments where A >> B (chrF++ gap > 10) ──")
        for a, b in a_much_better[:5]:
            print(f"\n  Δ={a['chrf']-b['chrf']:+.1f}  A={a['chrf']:.1f}  B={b['chrf']:.1f}")
            print(f"    SRC : {a['src'][:100]}")
            print(f"    REF : {a['ref'][:100]}")
            print(f"    A   : {a['hyp'][:100]}")
            print(f"    B   : {b['hyp'][:100]}")

    # ── Regressions introduced by B ────────────────────────────────────────────
    regressions = [(a, b) for a, b in paired
                   if a["score_bucket"] == "good" and b["score_bucket"] == "poor"]
    print(f"\n── Regressions (A=good → B=poor) : {len(regressions)}")
    for a, b in regressions[:3]:
        print(f"\n  SRC : {a['src'][:100]}")
        print(f"  A   : {a['hyp'][:100]}  (chrF={a['chrf']:.1f})")
        print(f"  B   : {b['hyp'][:100]}  (chrF={b['chrf']:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EN→AR NMT error analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags
    parser.add_argument("--run",     action="store_true",
                        help="Run inference and save per-segment results")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze saved segment file(s)")

    # --run arguments
    parser.add_argument("--model",    default=None,
                        help="Model path or HuggingFace Hub id")
    parser.add_argument("--dataset",  default="data/prepared/multiun/test",
                        help="Dataset path or HF Hub id  (default: data/prepared/multiun/test)")
    parser.add_argument("--n_samples", type=int, default=1500,
                        help="Number of samples to evaluate  (default: 1500)")
    parser.add_argument("--output",   default=None,
                        help="Where to save segment JSON  (default: segments/<model_slug>.json)")

    # --analyze arguments
    parser.add_argument("--input",   default=None,
                        help="Segment JSON file to analyze (model A)")
    parser.add_argument("--compare", default=None,
                        help="Second segment JSON file for head-to-head (model B)")
    parser.add_argument("--top_n",   type=int, default=10,
                        help="Number of worst/best segments to show  (default: 10)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.run and not args.analyze:
        print("Please specify --run, --analyze, or both. Use --help for usage.")
        return

    segments_a = None
    meta_a     = None

    # ── RUN ───────────────────────────────────────────────────────────────────
    if args.run:
        if not args.model:
            raise ValueError("--model is required for --run mode")

        # Auto-generate output path if not given
        if not args.output:
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model.split("/")[-1])
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"segments/{slug}_{ts}.json"

        segments_a = run_inference(args)
        meta_a     = {
            "model"   : args.model,
            "dataset" : args.dataset,
            "n_samples": len(segments_a),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

    # ── ANALYZE ───────────────────────────────────────────────────────────────
    if args.analyze:
        # If --run was also used, segments_a is already in memory
        if segments_a is None:
            if not args.input:
                raise ValueError("--input is required for --analyze mode (unless --run is also used)")
            meta_a, segments_a = load_segments(args.input)
        elif args.input:
            # --run + --analyze + explicit --input → analyze the explicit file
            meta_a, segments_a = load_segments(args.input)

        if args.compare:
            meta_b, segments_b = load_segments(args.compare)
            report_single(meta_a, segments_a, top_n=args.top_n)
            report_single(meta_b, segments_b, top_n=args.top_n)
            report_compare(meta_a, segments_a, meta_b, segments_b, top_n=args.top_n)
        else:
            report_single(meta_a, segments_a, top_n=args.top_n)

    print(f"\n{'='*60}\n Done.\n{'='*60}\n")


if __name__ == "__main__":
    main()