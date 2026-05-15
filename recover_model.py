import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbTokenizerFast,
)


def recover_model(checkpoint_path, base_model_path, output_path):
    print(f"--- Recovering Trimmed Model ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base (Trimmed) Path: {base_model_path}")

    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load configuration and determine architecture
    config = AutoConfig.from_pretrained(base_model_path)
    is_m2m = "m2m_100" in config.model_type.lower()

    # 2. Identify and Load Weights (Robust Search)
    # Check for various common weight file names used by Accelerate/FSDP
    candidates = [
        ckpt_path / "pytorch_model_fsdp.bin",
        ckpt_path / "pytorch_model.bin",
        ckpt_path / "model.safetensors",
    ]

    # Handle sharded files if present
    shards = sorted(ckpt_path.glob("pytorch_model-*.bin")) + sorted(
        ckpt_path.glob("model-*.safetensors")
    )

    weights_path = None
    state_dict = None

    for candidate in candidates:
        if candidate.exists():
            weights_path = candidate
            print(f"Found weights: {weights_path.name}")
            break

    if weights_path:
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_path))
        else:
            state_dict = torch.load(str(weights_path), map_location="cpu")
    elif shards:
        print(f"Found {len(shards)} sharded weight files. Consolidating...")
        state_dict = {}
        for shard in shards:
            shard_dict = (
                torch.load(shard, map_location="cpu")
                if shard.suffix == ".bin"
                else load_file(str(shard))
            )
            state_dict.update(shard_dict)
    else:
        # If nothing is found, list directory contents to help the user debug
        files = [f.name for f in ckpt_path.iterdir()]
        raise FileNotFoundError(
            f"Could not find weights in {checkpoint_path}.\n"
            f"Files present: {files}\n"
            f"Expected one of: pytorch_model_fsdp.bin, pytorch_model.bin, or shards."
        )

    # Clean FSDP/Accelerate prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        for prefix in ("_orig_mod.", "module.", "_fsdp_wrapped_module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        new_state_dict[name] = v

    # 3. Initialize and Save Model
    if is_m2m:
        model = M2M100ForConditionalGeneration(config)
    else:
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.load_state_dict(new_state_dict)
    model.save_pretrained(str(out_dir))

    # 4. Save Tokenizer
    print("Saving tokenizer...")
    if is_m2m:
        tokenizer = M2M100Tokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = NllbTokenizerFast.from_pretrained(base_model_path)
    tokenizer.save_pretrained(str(out_dir))

    # 5. CRITICAL: Copy the vocabulary mapping (Preserves trimmed vocab info)
    vocab_map = Path(base_model_path) / "vocab_id_map.json"
    if vocab_map.exists():
        shutil.copy(vocab_map, out_dir / "vocab_id_map.json")
        print(f"✓ Copied vocab_id_map.json (Essential for evaluation)")

    print(f"Successfully recovered model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to ckpt_epochX_stepY"
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to models/m2m100-trimmed"
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    recover_model(args.checkpoint, args.base_model, args.output)
