"""
prepare_dataset.py — Run once before training.
Loads, filters, and saves the dataset to disk so train_finetune.py
can load it instantly without touching the HF Hub again.

Usage:
    python prepare_dataset.py \\
        --dataset opus-100 \\
        --max_samples 200000 \\
        --out_dir data/prepared
"""

import argparse
from pathlib import Path

from src.data_loader import get_mixed_dataset, get_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and save dataset for training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="opus-100",
        choices=["opus-100", "multiun", "flores",  "mixed"],
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200_000,
        help="Maximum number of sentence pairs to keep.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/prepared",
        help="Directory to save the filtered dataset.",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip quality filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"\n📦 Loading dataset: {args.dataset}")
    if args.dataset == "mixed":
        raw_ds = get_mixed_dataset(filter = not args.no_filter)
    else:
        raw_ds = get_train_dataset(args.dataset, filter=not args.no_filter)

    total = len(raw_ds)
    num_samples = min(args.max_samples, total)
    ds = raw_ds.shuffle(seed=42).select(range(num_samples))

    print(f"✅ Selected {num_samples:,} / {total:,} samples")

    # Split into train / dev / test
    test_size  = min(2000, int(num_samples * 0.01))   # 1% or 2000 max
    dev_size   = min(2000, int(num_samples * 0.01))

    tmp        = ds.train_test_split(test_size=test_size, seed=42)
    tmp2       = tmp["train"].train_test_split(test_size=dev_size, seed=42)

    splits = {
        "train": tmp2["train"],
        "dev":   tmp2["test"],
        "test":  tmp["test"],
    }

    for name, split in splits.items():
        path = out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        split.save_to_disk(str(path))
        print(f"💾 {name:5s}: {len(split):,} samples → {path}")

    print(f"\nNow run:")
    print(f"  accelerate launch train_finetune.py --prepared_data {out_dir}/train")
    print(f"  python test.py --model_path <model> --dataset {out_dir}/test")

    
if __name__ == "__main__":
    main()
