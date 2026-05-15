"""
src/tokenizer_utils.py
Vocabulary-pruning utilities for Arabic–English NMT.
Fixed: Extraction occurs before layer resizing to avoid IndexError.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (BatchEncoding, PreTrainedModel,
                          PreTrainedTokenizerBase)

ID_MAP_FILENAME = "vocab_id_map.json"


# ─────────────────────────────────────────────────────────────────────────────
# RemappedTokenizer
# ─────────────────────────────────────────────────────────────────────────────


class RemappedTokenizer:
    """
    Wraps a PreTrainedTokenizer so that every call transparently
    maps old (original) token-IDs → new (trimmed) token-IDs on the
    way in, and new → old on the way out.
    """

    _LANG_TAG_FORMATS: dict[str, list[str]] = {
        "ar": ["arb_Arab", "ar_AR", "__ar__"],
        "en": ["eng_Latn", "en_XX",  "__en__"],
    }


    def __init__(self, tokenizer: PreTrainedTokenizerBase, id_map: dict[int, int]):
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_fwd", id_map)  # old → new
        object.__setattr__(self, "_rev", {v: k for k, v in id_map.items()})  # new → old

        # Build a fast lookup tensor for the forward direction
        max_old = max(id_map.keys()) + 1
        fwd_t = torch.full((max_old,), 0, dtype=torch.long)
        for old, new in id_map.items():
            fwd_t[old] = new
        object.__setattr__(self, "_fwd_tensor", fwd_t)

    # ── attribute pass-through ────────────────────────────────────────────────

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tokenizer"), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_tokenizer"), name, value)

    # ── remapping helpers ─────────────────────────────────────────────────────

    def _remap_ids(self, ids):
        """Map old IDs → new IDs (forward direction)."""
        fwd_t = object.__getattribute__(self, "_fwd_tensor")
        fwd = object.__getattribute__(self, "_fwd")
        if isinstance(ids, torch.Tensor):
            return fwd_t[ids.clamp(0, len(fwd_t) - 1)]
        elif isinstance(ids, list):
            if ids and isinstance(ids[0], list):
                return [[fwd.get(i, 0) for i in seq] for seq in ids]
            return [fwd.get(i, 0) for i in ids]
        return ids

    def _remap(self, encoding: BatchEncoding) -> BatchEncoding:
        if "input_ids" in encoding:
            encoding["input_ids"] = self._remap_ids(encoding["input_ids"])
        if "labels" in encoding:
            encoding["labels"] = self._remap_ids(encoding["labels"])
        return encoding

    def _unremap(self, ids):
        """Map new IDs → old IDs (reverse direction) for decoding."""
        rev = object.__getattribute__(self, "_rev")
        return [rev.get(i, i) for i in ids]

    # ── public interface ──────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        tok = object.__getattribute__(self, "_tokenizer")
        return self._remap(tok(*args, **kwargs))

    def encode(self, *args, **kwargs):
        tok = object.__getattribute__(self, "_tokenizer")
        return self._remap_ids(tok.encode(*args, **kwargs))

    def batch_decode(self, sequences, **kwargs):
        tok = object.__getattribute__(self, "_tokenizer")
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return tok.batch_decode([self._unremap(seq) for seq in sequences], **kwargs)

    def decode(self, token_ids, **kwargs):
        tok = object.__getattribute__(self, "_tokenizer")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return tok.decode(self._unremap(token_ids), **kwargs)

    def get_lang_id(self, lang_code: str) -> int:
        """
        Return the *remapped* ID for a language tag.

        Handles three tokenizer families:
        - M2M100Tokenizer      → has native get_lang_id()
        - NllbTokenizerFast    → uses convert_tokens_to_ids("arb_Arab")
        - MBart50TokenizerFast → uses convert_tokens_to_ids("ar_AR")
        """
        tok = object.__getattribute__(self, "_tokenizer")
        fwd = object.__getattribute__(self, "_fwd")

        # Fast path: M2M100 already has get_lang_id
        if hasattr(tok, "get_lang_id"):
            orig_id = tok.get_lang_id(lang_code)
            return fwd.get(orig_id, 0)

        # Fallback: probe known tag formats via convert_tokens_to_ids
        candidates = self._LANG_TAG_FORMATS.get(lang_code, [lang_code])
        unk_id = getattr(tok, "unk_token_id", None)

        for tag in candidates:
            try:
                tid = tok.convert_tokens_to_ids(tag)
                if tid is not None and tid != unk_id:
                    return fwd.get(tid, 0)
            except Exception:
                continue

        raise ValueError(
            f"Cannot resolve language code '{lang_code}' for "
            f"{type(tok).__name__}. Tried: {candidates}. "
            f"Add the correct tag to RemappedTokenizer._LANG_TAG_FORMATS."
        )

    def __len__(self):
        return len(object.__getattribute__(self, "_fwd"))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mandatory_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Collect special / language-tag token IDs that must never be pruned."""
    mandatory: set[int] = set()

    for attr in (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "sep_token_id",
    ):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            mandatory.add(val)

    vocab = tokenizer.get_vocab()
    for token, tid in vocab.items():
        t = token.strip()
        is_lang_tag = (
            (t.startswith("__") and t.endswith("__"))
            or (len(t) == 8 and t[3] == "_")
            or (len(t) == 5 and t[2] == "_" and t[3:].isupper())
        )
        if is_lang_tag:
            mandatory.add(tid)

    return mandatory


def _collect_corpus_token_ids(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int,
) -> set[int]:
    """Tokenise a random subset of the corpus and return all token IDs seen."""
    n = min(sample_size, len(dataset))
    subset = dataset.shuffle(seed=42).select(range(n))
    used: set[int] = set()
    for key in ("en", "ar"):
        texts = [ex["translation"][key] for ex in subset]
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        for ids in enc["input_ids"]:
            used.update(ids)
    return used


# ─────────────────────────────────────────────────────────────────────────────
# Main trimming entry-point
# ─────────────────────────────────────────────────────────────────────────────


def trim_vocab_to_lang_pair(
    model,
    tokenizer,
    dataset,
    output_dir,
    sample_size: int = 50_000,
    verbose: bool = True,
):
    """
    Prune *model* and *tokenizer* to the tokens actually used by the
    AR↔EN corpus, save everything to *output_dir*, and return
    ``(trimmed_model, RemappedTokenizer)``.

    Order of operations
    ───────────────────
    1. Decide which token IDs to keep.
    2. **Extract** the matching rows from the embedding / lm_head matrices
       *while they are still at their original size*.
    3. Call ``resize_token_embeddings`` to shrink the matrices.
    4. Write the extracted rows into the now-correctly-sized matrices.
    5. Persist model, tokenizer, and the id→id mapping.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Decide which IDs to keep ───────────────────────────────────────────
    corpus_ids = _collect_corpus_token_ids(dataset, tokenizer, sample_size)
    mandatory_ids = _mandatory_token_ids(tokenizer)
    keep_ids = sorted(corpus_ids | mandatory_ids)
    id_map = {old: new for new, old in enumerate(keep_ids)}
    new_vocab_size = len(keep_ids)

    if verbose:
        print(f"  [trim] Original vocab: {len(tokenizer)}")
        print(f"  [trim] Trimmed vocab:  {new_vocab_size}")

    # ── 2. Extract weights BEFORE resizing ───────────────────────────────────
    # Cloning here while the layers are still at their original size is the
    # key fix – indexing with keep_ids is safe because every id < original size.
    old_emb_weights = model.get_input_embeddings().weight.data[keep_ids].clone()

    old_lm_weights = None
    old_lm_bias = None
    if hasattr(model, "lm_head"):
        old_lm_weights = model.lm_head.weight.data[keep_ids].clone()
        if model.lm_head.bias is not None:
            old_lm_bias = model.lm_head.bias.data[keep_ids].clone()

    # ── 3. Resize (shrinks matrices in-place) ─────────────────────────────────
    model.resize_token_embeddings(new_vocab_size)

    # ── 4. Assign extracted rows to the resized layers ───────────────────────
    model.get_input_embeddings().weight.data = old_emb_weights
    if hasattr(model, "lm_head"):
        model.lm_head.weight.data = old_lm_weights
        if old_lm_bias is not None:
            model.lm_head.bias.data = old_lm_bias

    # ── 5. Update config ──────────────────────────────────────────────────────
    model.config.vocab_size = new_vocab_size
    model.config.pad_token_id = id_map.get(tokenizer.pad_token_id, 0)
    model.config.eos_token_id = id_map.get(tokenizer.eos_token_id, 1)
    model.tie_weights()

    # ── 6. Persist ────────────────────────────────────────────────────────────
    with open(out_path / ID_MAP_FILENAME, "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)

    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    return model, RemappedTokenizer(tokenizer, id_map)


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_remapped_tokenizer(
    model_path: str,
    base_tokenizer_cls=None,
    **tokenizer_kwargs,
) -> "RemappedTokenizer":
    """
    Load a RemappedTokenizer from a directory produced by trim_vocab_to_lang_pair.

    Parameters
    ----------
    model_path : str | Path
        Directory containing both the HF tokenizer files and vocab_id_map.json.
    base_tokenizer_cls : PreTrainedTokenizer class, optional
        The tokenizer class to use for loading (e.g. M2M100Tokenizer).
        If None, uses AutoTokenizer.
    **tokenizer_kwargs
        Forwarded to the tokenizer constructor (e.g. src_lang="en").

    Returns
    -------
    RemappedTokenizer
        Ready-to-use tokenizer that maps original ↔ trimmed IDs transparently.

    Raises
    ------
    FileNotFoundError
        If vocab_id_map.json is missing from model_path (untrimmed model).
    """
    model_path = Path(model_path)
    id_map_path = model_path / ID_MAP_FILENAME

    if not id_map_path.exists():
        raise FileNotFoundError(
            f"'{id_map_path}' not found. "
            f"Is '{model_path}' a trimmed model directory? "
            f"Run trim_vocab_to_lang_pair first."
        )

    # Load the base tokenizer (original vocab on disk — that's intentional)
    if base_tokenizer_cls is None:
        from transformers import AutoTokenizer

        base_tokenizer_cls = AutoTokenizer

    base_tok = base_tokenizer_cls.from_pretrained(str(model_path), **tokenizer_kwargs)

    # Reconstruct the forward id_map from the saved JSON
    with open(id_map_path) as f:
        raw = json.load(f)
    id_map = {int(k): v for k, v in raw.items()}

    return RemappedTokenizer(base_tok, id_map)


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_forced_bos_token_id(
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[int]:
    name = model_name_or_path.lower()
    if hasattr(tokenizer, "get_lang_id"):
        try:
            return tokenizer.get_lang_id("ar")
        except Exception:
            pass
    if "m2m100" in name:
        return tokenizer.get_lang_id("ar")
    if "mbart" in name:
        return tokenizer.lang_code_to_id["ar_AR"]
    if "nllb" in name:
        return tokenizer.convert_tokens_to_ids("arb_Arab")
    return None
