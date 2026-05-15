import logging

import evaluate
import numpy as np
from bert_score import score as bert_score_fn
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# ── Compatibility patch ───────────────────────────────────────────────────────
# Newer versions of transformers removed build_inputs_with_special_tokens from
# BertTokenizer (slow), but bert_score still calls it for empty-string inputs.
# We re-add the method so both libraries can coexist without upgrading either.
if not hasattr(BertTokenizer, "build_inputs_with_special_tokens"):

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return (
            [self.cls_token_id]
            + token_ids_0
            + [self.sep_token_id]
            + token_ids_1
            + [self.sep_token_id]
        )

    BertTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens

# ── Load scorers once at module level ────────────────────────────────────────
_bleu = evaluate.load("sacrebleu")
_ter = evaluate.load("ter")


def compute_metrics(preds: list[str], labels: list[str], lang: str = "ar") -> dict:
    """
    Computes a comprehensive set of NMT metrics:
      - SacreBLEU   : n-gram precision (standard for NMT papers)
      - TER         : Translation Edit Rate (lower is better)
      - BERTScore   : Contextual semantic similarity (F1)
      - Length Ratio: Avg predicted length / avg reference length
                      (flags systematic over/under-translation)

    Args:
        preds  : List of model-generated translations.
        labels : List of reference translations (one per source).
        lang   : BERTScore language code (default: 'ar').

    Returns:
        dict with keys: bleu, ter, bert_score_f1,
                        bert_score_p, bert_score_r, length_ratio
    """
    if not preds or not labels:
        raise ValueError("Predictions and labels must not be empty.")
    if len(preds) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(labels)} labels."
        )

    # ── Normalize ────────────────────────────────────────────────────────────
    decoded_preds = [p.strip() for p in preds]
    decoded_labels = [[l.strip()] for l in labels]  # list-of-lists for sacreBLEU
    flat_labels = [l[0] for l in decoded_labels]  # flat list for BERTScore / TER

    # ── SacreBLEU ────────────────────────────────────────────────────────────
    bleu_result = _bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    # ── TER (Translation Edit Rate) ──────────────────────────────────────────
    ter_result = _ter.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    # ── BERTScore ────────────────────────────────────────────────────────────
    P, R, F = bert_score_fn(
        cands=decoded_preds,
        refs=flat_labels,
        lang=lang,
        device="cuda",
        batch_size=64,
    )

    # ── Length Ratio ─────────────────────────────────────────────────────────
    pred_lengths = [len(p.split()) for p in decoded_preds]
    ref_lengths = [len(r.split()) for r in flat_labels]
    length_ratio = np.mean(pred_lengths) / (np.mean(ref_lengths) + 1e-9)

    results = {
        # Primary metrics
        "bleu": round(bleu_result["score"], 4),
        "ter": round(ter_result["score"], 4),
        # BERTScore (all three components)
        "bert_score_p": round(float(P.mean()), 4),
        "bert_score_r": round(float(R.mean()), 4),
        "bert_score_f1": round(float(F.mean()), 4),
        # Diagnostic
        "length_ratio": round(float(length_ratio), 4),
    }

    logger.info(
        "Metrics computed on %d samples | BLEU: %.2f | TER: %.2f | "
        "BERTScore-F1: %.4f | Length Ratio: %.3f",
        len(preds),
        results["bleu"],
        results["ter"],
        results["bert_score_f1"],
        results["length_ratio"],
    )

    return results
