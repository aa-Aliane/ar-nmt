# check_save.py
"""
Run after training to verify the saved checkpoint is healthy.
Usage: python check_save.py --model_path models/m2m100_finetuned_XXXXXXXX_XXXXXX
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from src.model_loader import load_nmt_model
from src.tokenizer_utils import ID_MAP_FILENAME, RemappedTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ref_model_path", default="models/m2m100-trimmed",
                        help="The trimmed model to compare weights against.")
    return parser.parse_args()


def check_state_dict_keys(model_path):
    print("\n" + "="*60)
    print("CHECK 1 — State dict keys")
    print("="*60)
    # Load the raw safetensors/bin without from_pretrained to see raw keys
    sd_path = Path(model_path)
    bin_files = list(sd_path.glob("*.bin")) + list(sd_path.glob("*.safetensors"))
    if not bin_files:
        print("✗ No weight files found!")
        return False

    if bin_files[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(bin_files[0]))
    else:
        sd = torch.load(str(bin_files[0]), map_location="cpu")

    sample_keys = list(sd.keys())[:5]
    print(f"  First 5 keys in checkpoint: {sample_keys}")

    has_orig_mod = any("_orig_mod" in k for k in sd.keys())
    if has_orig_mod:
        print("  ✗ KEYS STILL HAVE '_orig_mod.' PREFIX — Fix 1 did not work!")
        return False
    else:
        print("  ✓ Keys look clean (no _orig_mod prefix)")
        return True


def check_weights_changed(model_path, ref_model_path):
    print("\n" + "="*60)
    print("CHECK 2 — Weights actually changed vs trimmed baseline")
    print("="*60)
    finetuned = M2M100ForConditionalGeneration.from_pretrained(model_path)
    baseline  = M2M100ForConditionalGeneration.from_pretrained(ref_model_path)

    diffs = []
    for name, param in finetuned.named_parameters():
        if name in dict(baseline.named_parameters()):
            ref_param = dict(baseline.named_parameters())[name]
            diff = (param.float() - ref_param.float()).abs().mean().item()
            diffs.append((name, diff))

    diffs.sort(key=lambda x: -x[1])
    print(f"  Top 5 most-changed layers:")
    for name, diff in diffs[:5]:
        print(f"    {diff:.6f}  {name}")

    avg_diff = sum(d for _, d in diffs) / len(diffs)
    print(f"\n  Average weight diff vs baseline: {avg_diff:.6f}")
    if avg_diff < 1e-6:
        print("  ✗ Weights are IDENTICAL to baseline — model never trained or save failed")
        return False
    else:
        print("  ✓ Weights differ from baseline — training did something")
        return True


def check_translation(model_path):
    print("\n" + "="*60)
    print("CHECK 3 — Translation output")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_nmt_model(model_path, device=device)
    model.eval()

    test_sentences = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I want to learn Arabic.",
    ]

    lang_id = tokenizer.get_lang_id("ar")
    print(f"  forced_bos_token_id (ar, remapped): {lang_id}")

    for src in test_sentences:
        inputs = tokenizer(
            src, return_tensors="pt", truncation=True, padding=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=lang_id,
                max_new_tokens=64,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print(f"\n  SRC : {src}")
        print(f"  HYP : {decoded}")
        print(f"  (empty={len(decoded.strip()) == 0}, length={len(decoded)})")


def check_vocab_alignment(model_path):
    print("\n" + "="*60)
    print("CHECK 4 — Vocab / embedding alignment")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_nmt_model(model_path, device=device)

    embed_size = model.get_input_embeddings().weight.shape[0]
    vocab_size  = len(tokenizer)
    print(f"  Embedding rows : {embed_size}")
    print(f"  Tokenizer size : {vocab_size}")

    if embed_size != vocab_size:
        print("  ✗ MISMATCH — tokenizer and embeddings are out of sync")
        return False
    else:
        print("  ✓ Aligned")
        return True


def main():
    args = parse_args()
    print(f"\nDiagnosing: {args.model_path}")

    keys_ok    = check_state_dict_keys(args.model_path)
    weights_ok = check_weights_changed(args.model_path, args.ref_model_path)
    vocab_ok   = check_vocab_alignment(args.model_path)
    check_translation(args.model_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Keys clean      : {'✓' if keys_ok    else '✗'}")
    print(f"  Weights changed : {'✓' if weights_ok else '✗'}")
    print(f"  Vocab aligned   : {'✓' if vocab_ok   else '✗'}")


if __name__ == "__main__":
    main()




