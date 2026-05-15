"""
eval_model.py — Evaluate a fine-tuned NMT model on a prepared dataset.

Usage:
    python eval_model.py \
        --model_path models/m2m100_finetuned_20240415_120000 \
        --prepared_data data/prepared \
        --src_lang en \
        --tgt_lang ar \
        --batch_size 16
"""

import argparse
from pathlib import Path

import datasets as hf_datasets
import torch
from sacrebleu.metrics import BLEU, CHRF
from tqdm.auto import tqdm

from src.model_loader import load_nmt_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NMT model.")
    parser.add_argument("--model_path",    type=str, required=True,  help="Path to the fine-tuned model directory.")
    parser.add_argument("--prepared_data", type=str, required=True,  help="Path to dataset saved by prepare_dataset.py.")
    parser.add_argument("--src_lang",      type=str, default="en",   help="Source language code (default: en).")
    parser.add_argument("--tgt_lang",      type=str, default="ar",   help="Target language code (default: ar).")
    parser.add_argument("--batch_size",    type=int, default=16,      help="Inference batch size.")
    parser.add_argument("--max_length",    type=int, default=256,     help="Max tokens to generate.")
    parser.add_argument("--num_beams",     type=int, default=4,       help="Beam search width.")
    parser.add_argument("--max_samples",   type=int, default=100,    help="Cap evaluation to N samples (default: all).")
    return parser.parse_args()


def batch_translate(model, tokenizer, texts, src_lang, tgt_lang, max_length, num_beams, device, model_path):
    """Tokenize a batch of source sentences and return decoded translations."""
    inner_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    inner_tok.src_lang = src_lang  # e.g. "en"

    # RemappedTokenizer handles remapping internally
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Resolve the forced_bos_token_id through the remapped vocabulary
    import json
    from src.tokenizer_utils import ID_MAP_FILENAME
    id_map_path = Path(model_path) / ID_MAP_FILENAME
    if id_map_path.exists():
        with open(id_map_path) as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        original_bos = inner_tok.get_lang_id(tgt_lang)
        forced_bos = id_map.get(original_bos, original_bos)
    else:
        forced_bos = inner_tok.get_lang_id(tgt_lang)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=num_beams,
            max_length=max_length,
        )

    return inner_tok.batch_decode(generated, skip_special_tokens=True)


def run_eval():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n📥 Loading model from {args.model_path} ...")
    model, tokenizer = load_nmt_model(args.model_path, device=str(device))
    model.eval()
    print("✅ Model loaded.\n")

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    print(f"📂 Loading dataset from {args.prepared_data} ...")
    ds = hf_datasets.load_from_disk(args.prepared_data)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"✅ {len(ds)} samples to evaluate.\n")

    # ── 3. Run inference in batches ───────────────────────────────────────────
    hypotheses = []   # model outputs
    references  = []  # gold translations

    inner_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    inner_tok.src_lang = args.src_lang

    pbar = tqdm(range(0, len(ds), args.batch_size), desc="Translating")
    for start in pbar:
        batch = ds.select(range(start, min(start + args.batch_size, len(ds))))
        src_texts = [ex[args.src_lang] for ex in batch["translation"]]
        tgt_texts = [ex[args.tgt_lang] for ex in batch["translation"]]

        preds = batch_translate(
            model, tokenizer, src_texts,
            args.src_lang, args.tgt_lang, args.max_length, args.num_beams, device,
            model_path=args.model_path,
            )

        hypotheses.extend(preds)
        references.extend(tgt_texts)

    # ── 4. Score ──────────────────────────────────────────────────────────────
    bleu = BLEU()
    chrf = CHRF()

    # sacrebleu expects references as a list-of-lists
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])

    print("\n" + "─" * 50)
    print(f"  Model   : {args.model_path}")
    print(f"  Data    : {args.prepared_data}  ({len(ds)} samples)")
    print(f"  {args.src_lang} → {args.tgt_lang}")
    print("─" * 50)
    print(f"  BLEU  : {bleu_score.score:.2f}")
    print(f"  chrF  : {chrf_score.score:.2f}")
    print("─" * 50)

    # ── 5. Print a few examples ───────────────────────────────────────────────
    print("\n📋 Sample predictions (first 5):\n")
    src_samples = [ds[i]["translation"][args.src_lang] for i in range(min(5, len(ds)))]
    for i, (src, hyp, ref) in enumerate(zip(src_samples, hypotheses[:5], references[:5])):
        print(f"  [{i+1}] SRC : {src}")
        print(f"       HYP : {hyp}")
        print(f"       REF : {ref}")
        print()


if __name__ == "__main__":
    run_eval()