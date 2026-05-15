"""
comparaison.py
──────────────
Compare BLEU scores between the original M2M-100 model and a trimmed version.

Key fix
───────
The previous version called ``model.resize_token_embeddings(len(tokenizer))``
on the trimmed model.  The tokenizer saved to disk still carries the *original*
128 104-token SentencePiece vocab, so ``len(tokenizer) == 128 104``.
That one line silently re-expanded the trimmed embedding matrix back to
128 104 rows, filling the new rows with random noise → BLEU ≈ 0.

Additionally, ``tokenizer.batch_decode`` was called with the plain tokenizer,
which maps trimmed IDs (0-25 994) to the wrong original tokens, producing
garbage output.

Fix: for the trimmed model we reconstruct a ``RemappedTokenizer`` from the
saved ``vocab_id_map.json``, skip any resize, and route both
``forced_bos_token_id`` and ``batch_decode`` through it.
"""

import argparse
import json
import os
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from src.tokenizer_utils import ID_MAP_FILENAME, RemappedTokenizer

# ── Settings ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ORIGINAL = "facebook/m2m100_418M"
MODEL_TRIMMED = "./models/m2m100_finetuned_20260415_151238"
NUM_SAMPLES = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NMT model")
    parser.add_argument("--dataset", required=False)
    return parser.parse_args()



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_tokenizer(model_path: str):
    """
    Return the correct tokenizer for *model_path*.

    • Original model  → plain M2M100Tokenizer.
    • Trimmed model   → plain M2M100Tokenizer wrapped in RemappedTokenizer,
                        reconstructed from the saved vocab_id_map.json.
    """
    base_tok = M2M100Tokenizer.from_pretrained(model_path)
    base_tok.src_lang = "en"
    base_tok.tgt_lang = "ar"

    id_map_path = Path(model_path) / ID_MAP_FILENAME
    if id_map_path.exists():
        with open(id_map_path) as f:
            raw = json.load(f)
        id_map = {int(k): v for k, v in raw.items()}
        return RemappedTokenizer(base_tok, id_map)

    return base_tok


def _get_lang_id(tokenizer, lang: str = "ar") -> int:
    """Return the *tokenizer-correct* language token ID (remapped if needed)."""
    if isinstance(tokenizer, RemappedTokenizer):
        return tokenizer.get_lang_id(lang)  # already returns remapped ID
    try:
        return tokenizer.get_lang_id(lang)
    except Exception:
        return tokenizer.convert_tokens_to_ids(f"__{lang}__")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_model(model_path: str, dataset) -> float:
    print(f"\n--- Evaluating: {model_path} ---")

    if model_path.startswith("./") and not os.path.exists(model_path):
        print(f"Error: local path '{model_path}' not found.")
        return 0.0

    # ── Load tokenizer (RemappedTokenizer for trimmed, plain for original) ────
    tokenizer = _load_tokenizer(model_path)

    # ── Load model — NO resize needed; the saved config is already correct ────
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    model.eval()
    #
    # NOTE: do NOT call model.resize_token_embeddings() here.
    # • Original model: embeddings are already 128 104 — no change needed.
    # • Trimmed model:  embeddings are already at new_vocab_size (e.g. 25 995).
    #   Calling resize with the *original* tokenizer's length (128 104) would
    #   re-expand the matrix with random noise and destroy the model weights.
    #

    lang_id = _get_lang_id(tokenizer, "ar")

    metric = evaluate.load("sacrebleu")
    predictions = []
    references = []

    try:
        for i in tqdm(range(NUM_SAMPLES)):
            pair = dataset[i]["translation"]
            src_text = pair["en"]
            ref_text = [pair["ar"]]

            if isinstance(tokenizer, RemappedTokenizer):
                # Tokenise with the base tokenizer to get a pt tensor, then
                # remap input_ids into the trimmed ID space.
                # NOTE: apply_() is CPU-only, so we use a pre-built lookup
                # tensor (fwd_tensor) which works on any device.
                base_tok = object.__getattribute__(tokenizer, "_tokenizer")
                fwd_tensor = object.__getattribute__(tokenizer, "_fwd_tensor")
                inputs = base_tok(
                    src_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(DEVICE)
                fwd_tensor = fwd_tensor.to(DEVICE)
                inputs["input_ids"] = fwd_tensor[
                    inputs["input_ids"].clamp(0, len(fwd_tensor) - 1)
                ]
            else:
                inputs = tokenizer(
                    src_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(DEVICE)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=lang_id,
                    max_new_tokens=128,
                )

            # batch_decode via RemappedTokenizer un-maps IDs → original vocab
            decoded = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            predictions.append(decoded)
            references.append(ref_text)

        results = metric.compute(predictions=predictions, references=references)
        return results["score"]

    except Exception as exc:
        import traceback

        print(f"\n[ERROR] Evaluation failed for '{model_path}': {exc}")
        traceback.print_exc()
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────


def main():

    args = parse_args()

    if args.dataset:
        print(f"Loading Dataset ({args.dataset.split('/')[-1]} ar-en)...")
        try:
            ds_ar = load_dataset(args.dataset, "arb_Arab", split="devtest", streaming=True)
            ds_en = load_dataset(args.dataset, "eng_Latn", split="devtest", streaming=True)
      
            ds = [
                {"translation": {"ar": ar_row["text"], "en": en_row["text"]}}
                for ar_row, en_row in zip(ds_ar, ds_en)
            ]
            print(f"Successfully paired {len(ds)} samples.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return
    else:
        print("Loading Dataset (Opus-100 ar-en)...")
        try:
            ds = load_dataset(
                "Helsinki-NLP/opus-100", "ar-en", split=f"test[:{NUM_SAMPLES}]"
            )
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return

    score_orig = evaluate_model(MODEL_ORIGINAL, ds)
    score_trim = evaluate_model(MODEL_TRIMMED, ds)

    print("\n" + "=" * 40)
    print("      FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Original Model BLEU: {score_orig:.2f}")
    print(f"Trimmed Model BLEU:  {score_trim:.2f}")
    if score_orig > 0 and score_trim > 0:
        delta = score_trim - score_orig
        print(f"Performance Delta:   {delta:+.2f}  ({delta/score_orig*100:+.1f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
