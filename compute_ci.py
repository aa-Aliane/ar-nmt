"""
compute_ci.py — Bootstrap confidence intervals (95%) pour BLEU, chrF++, TER, COMET
Usage: python compute_ci.py --model path/vers/ton/modele --dataset multiun --n_boot 1000
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER

from src.data_loader import get_dataset
from src.model_loader import load_nmt_model

# ── Arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True,  help="Nom ou chemin du modèle")
parser.add_argument("--dataset", default="multiun", help="multiun | opus-100 | flores")
parser.add_argument("--n_boot",  type=int, default=1000)
parser.add_argument("--n_test",  type=int, default=1500, help="Nb de phrases du test set")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--use_comet", action="store_true", help="Inclure COMET (plus lent)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Charger le modèle ─────────────────────────────────────
print(f"\n[1/4] Chargement du modèle : {args.model}")
model, tokenizer = load_nmt_model(args.model, device)
model.eval()

# ── 2. Charger le test set ───────────────────────────────────
print(f"[2/4] Chargement du dataset : {args.dataset}")
ds = get_dataset(args.dataset, split="test")
ds = ds.select(range(min(args.n_test, len(ds))))

sources    = [ex["translation"]["en"] for ex in ds]
references = [ex["translation"]["ar"] for ex in ds]

# ── 3. Générer les traductions ───────────────────────────────
print(f"[3/4] Génération des traductions ({len(sources)} phrases)...")

# Récupère le token cible selon l'architecture
tgt_lang = tokenizer.tgt_lang
forced_bos = None
if hasattr(tokenizer, "lang_code_to_id"):          # M2M100 / NLLB
    forced_bos = tokenizer.lang_code_to_id[tgt_lang]
elif hasattr(tokenizer, "convert_tokens_to_ids"):  # mBART
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

hypotheses = []
for i in tqdm(range(0, len(sources), args.batch_size)):
    batch = sources[i : i + args.batch_size]
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    gen_kwargs = dict(num_beams=4, max_length=256)
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    hypotheses.extend(decoded)

# Sauvegarde (utile si tu veux relancer le CI sans regénérer)
with open("hypotheses.txt", "w") as f:
    f.write("\n".join(hypotheses))
with open("references.txt", "w") as f:
    f.write("\n".join(references))
print("  → hypotheses.txt et references.txt sauvegardés")

# ── 4. Bootstrap CI ──────────────────────────────────────────
print(f"\n[4/4] Bootstrap resampling ({args.n_boot} itérations)...")

bleu_m  = BLEU(effective_order=True)
chrf_m  = CHRF(word_order=2)
ter_m   = TER()

bleu_scores, chrf_scores, ter_scores = [], [], []

for _ in tqdm(range(args.n_boot)):
    idx = np.random.choice(len(hypotheses), len(hypotheses), replace=True)
    h = [hypotheses[i] for i in idx]
    r = [references[i] for i in idx]

    bleu_scores.append(bleu_m.corpus_score(h, [r]).score)
    chrf_scores.append(chrf_m.corpus_score(h, [r]).score)
    ter_scores.append(ter_m.corpus_score(h,  [r]).score)

def ci(scores):
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# ── Résultats ────────────────────────────────────────────────
print("\n" + "="*55)
print(f"  Modèle  : {args.model}")
print(f"  Dataset : {args.dataset}  |  N={len(hypotheses)}  |  Boot={args.n_boot}")
print("="*55)
for name, scores, arrow in [
    ("BLEU   ", bleu_scores, "↑"),
    ("chrF++ ", chrf_scores, "↑"),
    ("TER    ", ter_scores,  "↓"),
]:
    m, lo, hi = ci(scores)
    print(f"  {name} {arrow}  {m:.2f}  [95% CI: {lo:.2f} – {hi:.2f}]")

# ── COMET (optionnel) ────────────────────────────────────────
if args.use_comet:
    from comet import download_model, load_from_checkpoint
    print("\n  Chargement COMET...")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    comet_scores = []
    for _ in tqdm(range(args.n_boot), desc="COMET bootstrap"):
        idx = np.random.choice(len(hypotheses), len(hypotheses), replace=True)
        data = [{"src": sources[i], "mt": hypotheses[i], "ref": references[i]} for i in idx]
        result = comet_model.predict(data, batch_size=16, gpus=1)
        comet_scores.append(result.system_score)

    m, lo, hi = ci(comet_scores)
    print(f"  COMET   ↑  {m:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")

print("="*55)