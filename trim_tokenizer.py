"""
trim_tokenizer.py
─────────────────
Trim a multilingual NMT model's vocabulary to Arabic + English only.
Includes automatic verification to ensure model integrity after trimming.

Root-cause fix
──────────────
The verifier previously loaded a *fresh* M2M100Tokenizer from the saved
directory.  That tokenizer still contains the **original** 128 104-token
SentencePiece vocab, so calls such as ``get_lang_id("ar")`` returned the
original ID (e.g. 40 407), which is out of bounds for the trimmed embedding
matrix (size ≈ 25 995) → IndexError.

Fix: ``verify_trimmed_model`` now accepts an optional ``remapped_tokenizer``
argument.  When called from ``main()`` the already-built ``RemappedTokenizer``
is passed directly.  When called standalone (e.g. a post-hoc check) it
reconstructs the wrapper from the saved ``vocab_id_map.json``.

Supported models
─────────────────
  facebook/nllb-200-distilled-600M         (256 001 → ~12–18 K)
  facebook/m2m100_418M                     (128 112 → ~12–18 K)
  facebook/mbart-large-50-many-to-many-mmt (250 054 → ~18–24 K)

Usage
──────
  python trim_tokenizer.py \\
      --model facebook/m2m100_418M \\
      --output models/m2m100-trimmed \\
      --sample_size 150000
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (AutoTokenizer, M2M100ForConditionalGeneration,AutoModelForSeq2SeqLM,
                          M2M100Tokenizer, MBart50TokenizerFast,
                          MBartForConditionalGeneration, NllbTokenizerFast)

from src.data_loader import get_mixed_dataset
from src.tokenizer_utils import (ID_MAP_FILENAME, RemappedTokenizer,
                                 trim_vocab_to_lang_pair)

# ─────────────────────────────────────────────────────────────────────────────
TRIMMABLE_MODELS = {
    "facebook/nllb-200-distilled-600M",
    "facebook/m2m100_418M",
    "facebook/mbart-large-50-many-to-many-mmt",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim multilingual NMT tokenizer vocabulary to AR+EN only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the trimmed model.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50_000,
        help="Number of sentence pairs for coverage. Recommended for AR: 100k-200k.",
    )
    parser.add_argument(
        "--dataset",
        choices=["opus-100", "multiun", "mixed"],
        default="mixed",
        help="Dataset to sample from for coverage analysis.",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip quality filtering.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model / tokenizer loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_name: str, device):
    name = model_name.lower()

    if "nllb" in name:
        tokenizer = NllbTokenizerFast.from_pretrained(
            model_name, src_lang="eng_Latn", tgt_lang="arb_Arab"
        )
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)

    elif "m2m100" in name:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = "ar"
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)

    elif "mbart" in name:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "ar_AR"
        model = MBartForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)

    else:
        raise ValueError(
            f"'{model_name}' is not recognised. "
            f"Trimmable models: {sorted(TRIMMABLE_MODELS)}"
        )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading
# ─────────────────────────────────────────────────────────────────────────────


def load_corpus(dataset_choice: str, apply_filter: bool):
    from datasets import load_dataset

    if dataset_choice == "mixed":
        return get_mixed_dataset(filter=apply_filter)

    elif dataset_choice == "opus-100":
        ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train")
        if apply_filter:
            from src.data_loader import filter_dataset

            ds = filter_dataset(ds)
        return ds

    else:  # multiun
        ds = load_dataset("Helsinki-NLP/multiun", "ar-en", split="train[:200000]")
        if apply_filter:
            from src.data_loader import filter_dataset

            ds = filter_dataset(ds)
        return ds


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────


def verify_trimmed_model(output_dir, device, remapped_tokenizer=None):
    """
    Load the just-saved model and run basic sanity checks.

    Parameters
    ----------
    output_dir : str | Path
        The directory produced by ``trim_vocab_to_lang_pair``.
    device : torch.device
    remapped_tokenizer : RemappedTokenizer, optional
        When supplied (typical case: passed from ``main``), re-used directly.
        When omitted, reconstructed from the saved ``vocab_id_map.json`` so
        this function can also be called as a standalone post-hoc check.

    Why this matters
    ----------------
    ``tokenizer.save_pretrained`` writes the *original* SentencePiece vocab
    (e.g. 128 104 tokens).  Loading a plain M2M100Tokenizer from that
    directory therefore yields original token IDs.  Passing one of those IDs
    (e.g. 40 407 for "__ar__") as ``forced_bos_token_id`` into a model whose
    embedding matrix only has 25 995 rows → IndexError.

    The fix: always route IDs through a RemappedTokenizer so every ID is
    guaranteed to be < new_vocab_size.
    """
    print("\n" + "=" * 50)
    print(" STARTING AUTOMATIC VERIFICATION")
    print("=" * 50)

    try:
        output_dir = Path(output_dir)

        # ── Reconstruct RemappedTokenizer if not provided ─────────────────────
        if remapped_tokenizer is None:
            # Auto-detect the right tokenizer class from the saved config
            
            base_tok = AutoTokenizer.from_pretrained(str(output_dir))
            with open(output_dir / ID_MAP_FILENAME) as f:
                raw_map = json.load(f)
            id_map = {int(k): v for k, v in raw_map.items()}
            tokenizer = RemappedTokenizer(base_tok, id_map)
        else:
            tokenizer = remapped_tokenizer
            # We still need the underlying base tokenizer for raw tensor ops
            base_tok = object.__getattribute__(tokenizer, "_tokenizer")
            id_map = object.__getattribute__(tokenizer, "_fwd")

        # ── Load trimmed model ────────────────────────────────────────────────
        model = AutoModelForSeq2SeqLM.from_pretrained(str(output_dir)).to(device)

        # ── Check 1: vocab size ───────────────────────────────────────────────
        trimmed_size = len(tokenizer)
        print(f" [√] New Vocab Size: {trimmed_size}")

        # ── Check 2: unknown-token rate ───────────────────────────────────────
        test_sentence = (
            "The financial market is unstable. " "This is a verification test."
        )
        remapped_ids = tokenizer.encode(test_sentence)  # already in new ID space
        unk_new_id = id_map.get(base_tok.unk_token_id, 0)
        unk_count = remapped_ids.count(unk_new_id)
        if unk_count > 0:
            print(
                f" [!] WARNING: Found {unk_count} <unk> token(s) in test sentence.\n"
                "     Consider increasing --sample_size for better coverage."
            )
        else:
            print(" [√] Vocabulary Coverage: Success (0 <unk> tokens).")

        # ── Check 3: end-to-end translation ───────────────────────────────────
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = "ar"

        # Tokenise with the base tokenizer, then remap IDs into new space.
        # apply_() is CPU-only — use the pre-built fwd_tensor lookup instead.
        fwd_tensor = object.__getattribute__(tokenizer, "_fwd_tensor").to(device)
        raw_enc = base_tok(test_sentence, return_tensors="pt").to(device)
        raw_enc["input_ids"] = fwd_tensor[
            raw_enc["input_ids"].clamp(0, len(fwd_tensor) - 1)
        ]

        # forced_bos_token_id must be a *remapped* ID
        lang_id = tokenizer.get_lang_id("ar")  # RemappedTokenizer handles this
        outputs = model.generate(**raw_enc, forced_bos_token_id=lang_id)

        # batch_decode via RemappedTokenizer un-maps IDs back before decoding
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if len(decoded.strip()) < 5:
            print(" [X] ERROR: Model produced empty or invalid Arabic output.")
        else:
            print(f" [√] Translation Test: Success")
            print(f"     Output: {decoded}")

    except Exception as exc:
        import traceback

        print(f" [X] VERIFICATION CRITICAL ERROR: {exc}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    device = (
        torch.device("cpu")
        if args.cpu
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    print(f"\n Device : {device}")

    # 1. Load original model + tokenizer
    print(f" Loading model : {args.model} …")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # 2. Load corpus
    dataset = load_corpus(args.dataset, apply_filter=not args.no_filter)

    # 3. Trim
    print(f" Trimming vocabulary using {args.sample_size} samples...")
    model, remapped_tok = trim_vocab_to_lang_pair(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output,
        sample_size=args.sample_size,
        verbose=True,
    )

    # 4. Verify — pass the already-built RemappedTokenizer so the verifier
    #    never touches the original-vocab tokenizer on disk.
    verify_trimmed_model(args.output, device, remapped_tokenizer=remapped_tok)

    print(f"\n Done. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
