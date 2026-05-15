import argparse
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

RESULTS_DIR = Path("results")
MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_BEAMS = 4
DATASETS = ["opus-100", "multiun"]


def load_finetuned_model(model_path: str, model_base: str, accelerator):
    """
    Load model and tokenizer ensuring they are placed on the correct local GPU.
    """
    device = accelerator.device
    print(f"  [Rank {accelerator.process_index}] Loading model from: {model_path}")

    # Load model and tokenizer using the custom project loader
    # We pass the specific device to prevent 'device_map="auto"' conflicts
    model, tokenizer = load_nmt_model(model_path, device=device)

    base = model_base.lower()
    forced_bos = None

    if "m2m100" in base:
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = "ar"
        # CRITICAL: We use convert_tokens_to_ids because the RemappedTokenizer
        # ensures this returns the NEW index in the trimmed embedding layer.
        forced_bos = tokenizer.convert_tokens_to_ids("ar")

    elif "nllb" in base:
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.src_lang = "eng_Latn"
            tokenizer.tokenizer.tgt_lang = "arb_Arab"
        else:
            tokenizer.src_lang = "eng_Latn"
            tokenizer.tgt_lang = "arb_Arab"
        forced_bos = tokenizer.convert_tokens_to_ids("arb_Arab")

    # Move model to device and prepare for inference
    model.eval()
    # We don't necessarily need accelerator.prepare(model) for pure inference
    # if it's already on the correct device, but it helps ensure consistency.
    model = accelerator.prepare(model)

    return model, tokenizer, forced_bos


def collate_fn(batch):
    sources = [ex["translation"]["en"] for ex in batch]
    targets = [ex["translation"]["ar"] for ex in batch]
    return sources, targets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned NMT model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_base", required=True, choices=["m2m100", "mbart", "nllb", "opus"]
    )
    parser.add_argument("--dataset", default=None, choices=DATASETS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_to_eval = [args.dataset] if args.dataset else DATASETS

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    is_main = accelerator.is_main_process

    # Label for logging
    label = Path(args.model_path).name

    if is_main:
        print(f"\n{'='*62}")
        print(f"  Model      : {label}")
        print(f"  Base arch  : {args.model_base}")
        print(f"  Processes  : {accelerator.num_processes}")
        print(f"{'='*62}")

    # Load Model (Handles Device Placement & Tokenizer Remapping)
    model, tokenizer, forced_bos = load_finetuned_model(
        args.model_path, args.model_base, accelerator
    )

    all_results = {}

    for ds_name in datasets_to_eval:
        if is_main:
            print(f"\n  Dataset: {ds_name}")

        try:
            ds = get_dataset(ds_name)
        except Exception as e:
            if is_main:
                print(f"  [SKIP] {e}")
            continue

        dataloader = DataLoader(
            ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2
        )
        # Distribute the dataloader across GPUs
        dataloader = accelerator.prepare(dataloader)

        all_preds, all_labels = [], []

        for sources, targets in tqdm(
            dataloader, desc=f"Eval {ds_name}", disable=not is_main
        ):
            # Tokenize and move to the current process device
            inputs = tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(accelerator.device)

            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "early_stopping": True,
            }
            if forced_bos is not None:
                gen_kwargs["forced_bos_token_id"] = forced_bos

            with torch.no_grad():
                # For M2M100/NLLB, using the prepared model's generate
                generated_ids = accelerator.unwrap_model(model).generate(**gen_kwargs)

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Gather predictions from all GPUs back to the main process
            gathered_preds = gather_object(decoded)
            gathered_labels = gather_object(list(targets))

            if is_main:
                all_preds.extend(gathered_preds)
                all_labels.extend(gathered_labels)

        if is_main and all_preds:
            metrics = compute_metrics(all_preds, all_labels)
            all_results[ds_name] = metrics
            print(
                f"\n  Results [{ds_name}]: BLEU: {metrics['bleu']:.2f} | TER: {metrics['ter']:.2f}"
            )

    if is_main:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (
            Path(args.output)
            if args.output
            else RESULTS_DIR / f"eval_{label}_{run_id}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({label: all_results}, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved results to {out_path}")


if __name__ == "__main__":
    main()
