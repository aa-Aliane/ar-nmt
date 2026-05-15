"""
train_finetune.py — Multi-GPU fine-tuning with Accelerate.
Expects a dataset already prepared by prepare_dataset.py.

Usage:
    accelerate launch --num_processes 4 --mixed_precision bf16 \\
        train_finetune.py \\
        --base_model models/m2m100-trimmed \\
        --prepared_data data/prepared

Optimizations applied
──────────────────────
1. bf16 instead of fp16      → removes the _convert_to_fp32 OOM crash entirely
2. gradient_checkpointing    → cuts activation memory ~65% (trades compute)
3. 8-bit AdamW (bitsandbytes)→ cuts optimizer state memory ~75%
4. batch=4, grad_accum=4     → effective batch of 64 across 4 GPUs
5. torch.compile()           → fuses ops, reduces intermediate allocations
6. find_unused_parameters=False + bucket_cap_mb tuned for checkpointing
7. PYTORCH_CUDA_ALLOC_CONF   → reduces fragmentation
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import datasets as hf_datasets
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (DataCollatorForSeq2Seq,
                          get_linear_schedule_with_warmup)

# 8-bit optimizer — install with: pip install bitsandbytes
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("⚠️  bitsandbytes not found — falling back to AdamW fp32. "
          "Install with: pip install bitsandbytes", file=sys.stderr)
    from torch.optim import AdamW

# Local project imports
from src.model_loader import load_nmt_model
from src.tokenizer_utils import ID_MAP_FILENAME, RemappedTokenizer

MAX_GRAD_NORM = 1.0

# ── Reduce fragmentation (same as PyTorch docs recommend in OOM messages) ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trimmed NMT model.")
    parser.add_argument("--base_model", type=str, default="models/m2m100-trimmed")
    parser.add_argument(
        "--prepared_data",
        type=str,
        required=True,
        help="Path to dataset saved by prepare_dataset.py",
    )
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,                          # ← raised from 2; grad_accum keeps peak low
        help="Per-GPU batch size.",
    )
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,                          # effective batch = 4 × 4 × 4 GPUs = 64
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile() (useful for debugging).",
    )
    return parser.parse_args()


def _remapped_pad_token_id(base_model_path: str, inner_tok) -> int:
    """
    Return the *remapped* pad token ID so DataCollatorForSeq2Seq pads
    decoder_input_ids with the correct value for the trimmed vocabulary.
    """
    id_map_path = Path(base_model_path) / ID_MAP_FILENAME
    if not id_map_path.exists():
        return inner_tok.pad_token_id

    with open(id_map_path) as f:
        raw = json.load(f)
    id_map = {int(k): v for k, v in raw.items()}
    return id_map.get(inner_tok.pad_token_id, inner_tok.pad_token_id)


def run_finetune():
    args = parse_args()

    # Avoid NCCL P2P issues on multi-GPU consumer cards
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    # ── 1. Accelerator setup ──────────────────────────────────────────────────
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))

    # find_unused_parameters=True is required when gradient_checkpointing is
    # enabled: checkpointing breaks the normal backward graph so DDP sees some
    # params as "unused" even though they are not.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        bucket_cap_mb=25,               # smaller buckets → lower peak allreduce memory
    )

    accelerator = Accelerator(
        mixed_precision="bf16",         # ← bf16 replaces fp16; no loss scaler needed,
                                        #   eliminates the _convert_to_fp32 OOM crash
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[process_group_kwargs, ddp_kwargs],
    )
    is_main = accelerator.is_main_process

    if not is_main:
        hf_datasets.disable_progress_bar()

    if is_main:
        print(f"\n🚀 Starting Fine-tuning Pipeline")
        print(f"   GPUs            : {accelerator.num_processes}")
        print(f"   Mixed precision : bf16")
        print(f"   Batch / GPU     : {args.batch_size}")
        print(f"   Grad accum      : {args.grad_accum_steps}")
        print(f"   Effective batch : {args.batch_size * args.grad_accum_steps * accelerator.num_processes}")
        print(f"   8-bit AdamW     : {HAS_BNB}")
        print(f"   torch.compile   : {not args.no_compile}")

    # ── 2. Model loading ──────────────────────────────────────────────────────
    if is_main:
        print("📥 [1/3] Loading model...")

    with accelerator.main_process_first():
        model, tokenizer = load_nmt_model(args.base_model, device="cpu")

    # Silence the tied-weights warning: the trimmed model intentionally keeps
    # separate embedding matrices for shared / encoder / decoder / lm_head.
    model.config.tie_word_embeddings = False

    # use_cache is incompatible with gradient checkpointing
    model.config.use_cache = False

    # ── Optimization 1: gradient checkpointing ────────────────────────────────
    # use_reentrant=False is the modern, memory-safer variant and avoids a
    # deprecation warning in PyTorch ≥ 2.1.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── Optimization 5: torch.compile ─────────────────────────────────────────
    # Must be done BEFORE accelerator.prepare(). Fuses ops → less intermediate
    # memory and faster throughput. Skip with --no_compile for debugging.
    if not args.no_compile:
        if is_main:
            print("🔧 Compiling model with torch.compile()...")
        model = torch.compile(model)

    if is_main:
        print("✅ Model loaded.")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    if is_main:
        print(f"📂 [2/3] Loading prepared dataset from {args.prepared_data} ...")

    ds = hf_datasets.load_from_disk(args.prepared_data)

    if is_main:
        print(f"✅ Dataset loaded: {len(ds)} samples")
        print("🔤 Tokenizing...")

    def preprocess_fn(examples):
        inputs  = [ex["en"] for ex in examples["translation"]]
        targets = [ex["ar"] for ex in examples["translation"]]

        inner_tok = getattr(tokenizer, "_tokenizer", tokenizer)
        inner_tok.src_lang = "en"
        inner_tok.tgt_lang = "ar"

        model_inputs = tokenizer(
            inputs, max_length=128, truncation=True, padding=False
        )
        labels = tokenizer(
            text_target=targets, max_length=128, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        tokenized_ds = ds.map(
            preprocess_fn,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )

    if is_main:
        print("✅ Tokenization complete.")

    # ── 4. DataLoader ─────────────────────────────────────────────────────────
    inner_tok  = getattr(tokenizer, "_tokenizer", tokenizer)
    remapped_pad = _remapped_pad_token_id(args.base_model, inner_tok)
    inner_tok.pad_token_id = remapped_pad       # fix: remap pad id for trimmed vocab

    collator = DataCollatorForSeq2Seq(
        inner_tok,
        model=model,
        padding=True,
        label_pad_token_id=-100,                # -100 is ignored by cross-entropy
    )

    dataloader = DataLoader(
        tokenized_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=True,
        pin_memory=True,                        # faster CPU→GPU transfers
        num_workers=2,                          # parallel data loading
    )

    # ── 5. Optimizer ──────────────────────────────────────────────────────────
    # Optimization 3: 8-bit AdamW cuts optimizer state memory by ~75%.
    # Adam keeps momentum + variance buffers = 2× model size in fp32 normally.
    # bnb.optim.AdamW8bit stores those in 8-bit → ~4× smaller.
    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
        if is_main:
            print("✅ Using 8-bit AdamW optimizer.")
    else:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = (len(dataloader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),     # 5% warmup
        num_training_steps=total_steps,
    )

    if is_main:
        print(f"\n⚙️  [3/3] Distributing across {accelerator.num_processes} GPU(s)...")
        print(f"   Batches/epoch : {len(dataloader)}")
        print(f"   Total steps   : {total_steps}")
        print(f"   Warmup steps  : {max(1, total_steps // 20)}")
        print(f"   Grad clip     : {MAX_GRAD_NORM}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if is_main:
        print("✅ Ready. Starting training...\n")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        # Clear fragmented cache between epochs
        torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            dataloader,
            disable=not is_main,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss    = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg":  f"{epoch_loss / (step + 1):.4f}",
            })

        if is_main:
            avg = epoch_loss / len(dataloader)
            print(f"   Epoch {epoch + 1}/{args.epochs} done — avg loss: {avg:.4f}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()

    state_dict = accelerator.get_state_dict(model)

    if is_main:
        out_dir = Path(
            f"models/m2m100_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)
        if hasattr(unwrapped, '_orig_mod'):
            unwrapped = unwrapped._orig_mod

        # Strip _orig_mod prefix
        clean_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

        # Force-add lm_head.weight if missing (deduplicated out by accelerate).
        if "lm_head.weight" not in clean_state_dict:
            clean_state_dict["lm_head.weight"] = clean_state_dict["model.shared.weight"].clone()

        # safetensors refuses to save tensors that share memory.
        # model.shared.weight / lm_head.weight / encoder.embed_tokens.weight /
        # decoder.embed_tokens.weight are all originally tied → same data_ptr.
        # Clone every one of them to break the sharing before writing to disk.
        tied_keys = [
            "model.shared.weight",
            "lm_head.weight",
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
        ]
        for key in tied_keys:
            if key in clean_state_dict:
                clean_state_dict[key] = clean_state_dict[key].clone()

        from safetensors.torch import save_file
        save_file(clean_state_dict, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

        # Save config and tokenizer normally
        unwrapped.config.tie_word_embeddings = False
        unwrapped.config.save_pretrained(out_dir)
        inner_tok.save_pretrained(out_dir)

        for fname in [ID_MAP_FILENAME, "sentencepiece.bpe.model"]:
            src = Path(args.base_model) / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

        print(f"\n✅ Training complete → {out_dir}")
        print(f"   Load with: load_nmt_model('{out_dir}')")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    run_finetune()