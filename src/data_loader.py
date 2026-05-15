"""
Data loading utilities for NMT evaluation and fine-tuning.

Additions over original:
  - filter_dataset()   : removes low-quality pairs (length ratio, min tokens)
  - get_mixed_dataset(): combines opus-100 + multiun into one shuffled dataset
"""

from datasets import Dataset, concatenate_datasets, load_dataset


# ──────────────────────────────────────────────────────────────
# Quality filter
# ──────────────────────────────────────────────────────────────
def filter_dataset(ds, min_tokens: int = 5, max_ratio: float = 2.0):
    """
    Remove low-quality sentence pairs:
      - Either side shorter than min_tokens words
      - Length ratio (longer / shorter) exceeds max_ratio

    Applied on raw (un-tokenized) text so it works before any tokenizer.
    """

    def is_good(example):
        en = example["translation"]["en"].strip()
        ar = example["translation"]["ar"].strip()
        
        en_words = en.split()
        ar_words = ar.split()

        # Word count check
        if len(en_words) < min_tokens or len(ar_words) < min_tokens:
            return False

        # Use CHARACTER ratio instead — much fairer across scripts
        char_ratio = max(len(en), len(ar)) / (min(len(en), len(ar)) + 1e-9)
        if char_ratio > 3.5:
            return False

        # Keep word ratio loose for AR-EN
        word_ratio = max(len(en_words), len(ar_words)) / (min(len(en_words), len(ar_words)) + 1e-9)
        return word_ratio <= 3.0

    original = len(ds)
    ds = ds.filter(is_good, num_proc=4, desc="Filtering low-quality pairs")
    kept = len(ds)
    print(
        f"  [filter] Kept {kept:,} / {original:,} pairs "
        f"({100*kept/original:.1f}%) after quality filtering"
    )
    return ds


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────
def get_dataset(name, split="test"):
    """
    Load a test/eval dataset split.
    Used by eval scripts — no filtering applied (eval on raw data).
    """
    if "opus" in name.lower():
        return load_dataset("Helsinki-NLP/opus-100", "ar-en", split=split)
    elif "multiun" in name.lower():
        return load_dataset("Helsinki-NLP/multiun", "ar-en", split="train[-10000:]")
    elif "flores" in name.lower():
        
        ds_ar = load_dataset("openlanguagedata/flores_plus", "arb_Arab", split="devtest", streaming=True)
        ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest", streaming=True)
    
        paired = [
            {"translation": {"ar": ar_row["text"], "en": en_row["text"]}}
            for ar_row, en_row in zip(ds_ar, ds_en)
        ]

        return Dataset.from_list(paired)

    else:
        raise ValueError(f"Dataset '{name}' not recognized. Choose: opus-100, multiun, flores")


def get_train_dataset(
    ds_name: str,
    filter: bool = True,
    min_tokens: int = 5,
    max_ratio: float = 2.0,
    max_samples: int | None = None,
):
    """
    Load a single training dataset, with optional quality filtering.

    Args:
        ds_name    : 'opus-100' or 'multiun'
        filter     : apply quality filter (default True)
        min_tokens : minimum words per side (default 5)
        max_ratio  : max length ratio longer/shorter side (default 2.0)
        max_samples: cap the dataset size (None = use all)
    """
    if "opus" in ds_name.lower():
        split = "train" if max_samples is None else f"train[:{max_samples}]"
        ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split=split)
    elif "multiun" in ds_name.lower():
        n = max_samples or 200_000
        ds = load_dataset("Helsinki-NLP/multiun", "ar-en", split=f"train[:{n}]")
    elif "flores" in ds_name.lower():       
        ds_ar = load_dataset("openlanguagedata/flores_plus", "arb_Arab", split="dev", streaming=True)
        ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="dev", streaming=True)
    
        paired = [
            {"translation": {"ar": ar_row["text"], "en": en_row["text"]}}
            for ar_row, en_row in zip(ds_ar, ds_en)
        ]

        return Dataset.from_list(paired)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    if filter:
        ds = filter_dataset(ds, min_tokens=min_tokens, max_ratio=max_ratio)

    return ds


def get_mixed_dataset(
    filter: bool = True,
    min_tokens: int = 5,
    max_ratio: float = 3.0,
    multiun_samples: int = 200_000,
    subtitles_samples: int = 200_000,
):
    """
    Combine opus-100 (train) + multiun (first multiun_samples rows),
    apply quality filtering to each, then shuffle and return.

    Mixing gives the model diverse sentence styles:
      - opus-100  : varied domains (news, books, subtitles, web)
      - multiun   : formal UN diplomatic text
    """

    print("  [data] Loading opus-100 train split...")
    opus = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train")

    print(f"  [data] Loading multiun train[:{multiun_samples}]...")
    multiun = load_dataset(
        "Helsinki-NLP/multiun", "ar-en", split=f"train[:{multiun_samples}]"
    )



    if filter:
        print("  [data] Filtering opus-100...")
        opus = filter_dataset(opus, min_tokens=min_tokens, max_ratio=max_ratio)
        print("  [data] Filtering multiun...")
        multiun = filter_dataset(multiun, min_tokens=min_tokens, max_ratio=max_ratio)


    combined = concatenate_datasets([opus, multiun]).shuffle(seed=42)
    print(
        f"  [data] Mixed dataset: {len(combined):,} total pairs "
        f"({len(opus):,} opus + {len(multiun):,} multiun"
    )
    return combined