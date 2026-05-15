# %%
import pandas as pd

# %%
"""
French MedCLIP Fine-tuning Pipeline — CASIA-CXR
=================================================
Replaces BioClinicalBERT with CamemBERT-bio (French biomedical encoder).
Keeps MedCLIP's Swin Transformer vision encoder and semantic matching loss.
Runs on 4 × RTX 2080 Ti (11 GB VRAM) using DDP + fp16 + all_gather.

File structure expected:
    project/
        data/
            train.csv
            val.csv
            test.csv
        train.py          ← this file
        checkpoints/      ← saved automatically

Usage:
    # Single GPU (debug)
    python train.py --debug

    # 4 GPUs (production)
    torchrun --nproc_per_node=4 train.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import math
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    SwinModel,
    SwinConfig,
)
from torchvision import transforms

# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# %%

class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    train_csv   = "data/train.csv"
    val_csv     = "data/val.csv"
    test_csv    = "data/test.csv"
    output_dir  = "checkpoints"

    # ── Text encoder — French biomedical ────────────────────────────────────
    # Option A (recommended first — lighter, good NER performance)
    text_encoder_name = "almanach/camembert-bio-base"
    # Option B (stronger on clinical text per DrBenchmark — comment A, uncomment B)
    # text_encoder_name = "Dr-BERT/DrBERT-7GB"

    # ── Vision encoder — MedCLIP's Swin Transformer ─────────────────────────
    # We load pretrained MedCLIP weights → keeps the visual encoder aligned
    # with the original medical embedding space
    vision_encoder_name = "microsoft/swin-base-patch4-window7-224-in22k"

    # ── Tokenizer settings ────────────────────────────────────────────────────
    max_length  = 128      # MedCLIP default is 77; French reports are longer → 128

    # ── Image settings ────────────────────────────────────────────────────────
    image_size  = 224      # Swin-B expects 224×224

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          = 30
    batch_size      = 32       # Per GPU — effective = 32 × 4 = 128
    lr              = 5e-5     # Lower than MedCLIP default (1e-4) — we're fine-tuning
    weight_decay    = 1e-4
    warmup_epochs   = 2
    grad_clip       = 1.0

    # ── Projection head ───────────────────────────────────────────────────────
    embed_dim   = 512      # MedCLIP paper: output dim 512
    temperature = 0.07     # MedCLIP paper: learnable τ initialized at 0.07

    # ── Which encoder layers to freeze ───────────────────────────────────────
    # Freeze bottom N layers to prevent catastrophic forgetting on 11K samples
    freeze_text_layers   = 8    # CamemBERT-bio has 12 layers → freeze bottom 8
    freeze_vision_layers = 2    # Swin has 4 stages → freeze bottom 2 stages

    # ── Report sections to use ────────────────────────────────────────────────
    # Based on MedCLIP paper: Findings + Impression
    # We add Indication for French clinical context
    text_sections = ["Findings", "Impression", "Indication"]

    # ── Hardware ─────────────────────────────────────────────────────────────
    num_workers = 4
    pin_memory  = True
    seed        = 42


cfg = Config()

# %%
def build_report_text(row: pd.Series) -> str:
    """
    Combines report sections into one clean French string.

    Decisions based on prior work:
    - Findings + Impression: from MedCLIP paper (combines both sections)
    - Indication added: useful French clinical context
    - Comparison excluded: references prior studies often absent in CASIA-CXR
    - Keep absence/negation phrases: clinically meaningful, BERT handles them
    - "/" replaced with ". ": CASIA-CXR uses "/" as sentence separator
    - Lowercase: CamemBERT-bio/DrBERT trained on lowercased text
    - No stemming/lemmatization: destroys subword tokenization quality
    - Sentences < 3 words removed: from MedCLIP paper
    """
    section_labels = {
        "Findings":   "résultats",
        "Impression": "impression",
        "Indication": "indication",
    }

    parts = []
    for col, label in section_labels.items():
        val = str(row.get(col, "")).strip()

        # Drop empty, NaN, and the CASIA-CXR "#" end marker
        if not val or val in ("nan", "#", ""):
            continue

        # Replace "/" sentence separator with ". "
        val = val.replace("/", "")

        # Normalise whitespace
        val = re.sub(r"\s+", " ", val).strip()

        # Lowercase
        val = val.lower()

        # Split into sentences and filter those with < 3 words (MedCLIP paper)
        sentences = [s.strip() for s in val.split(".") if s.strip()]
        sentences = [s for s in sentences if len(s.split()) >= 3]

        if sentences:
            section_text = ". ".join(sentences) + "."
            parts.append(f"{label}: {section_text}")

    return " ".join(parts)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing and drop unusable rows."""
    df = df.copy()
    df["report_text"] = df.apply(build_report_text, axis=1)

    

    before = len(df)
    # Drop rows where resulting text is too short
    df = df[df["report_text"].str.len() > 20].reset_index(drop=True)
    after = len(df)

    if before - after > 0:
        log.warning(f"Dropped {before - after} rows with insufficient report text.")

    total      = len(df)
    n_unique   = df["report_text"].nunique()
    dup_rate   = 1 - n_unique / total
    top5       = df["report_text"].value_counts().head(5)
    
    print(f"\n{'='*50}")
    print(f" {n_unique} unique / {total} total ({dup_rate:.1%} duplicates)")
    print(f"Top 5 most repeated reports:")
    for text, count in top5.items():
        print(f"  [{count}×] {text[:80]}...")

    for i in range(5):
        print("\n--- PROCESSED ---")
        print(df.iloc[i]["report_text"])

    return df

# %%
def apply_clahe(pil_image: Image.Image) -> Image.Image:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Normalises local contrast in X-rays from different machines/hospitals.
    Applied before other transforms on the raw grayscale image.
    clipLimit=2.0, tileGridSize=(8,8) — standard settings for chest X-rays.
    """
    gray = np.array(pil_image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Convert back to PIL RGB (3-channel repeat — X-rays are grayscale)
    return Image.fromarray(enhanced).convert("RGB")


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Image transforms matching MedCLIP + ConVIRT augmentation strategy.

    Training: CLAHE → resize → random crop → flip → affine → color jitter
    Val/Test: CLAHE → resize → center crop → normalize (deterministic)

    Notes:
    - No saturation/hue jitter (X-rays are grayscale)
    - Small rotation only (±5°) to avoid cropping lung fields
    - ImageNet mean/std: standard for Swin pretrained on ImageNet
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Lambda(apply_clahe),
            transforms.Resize(256),
            transforms.RandomCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(apply_clahe),
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# %%
class CASIACXRDataset(Dataset):
    """
    PyTorch Dataset for MedCLIP fine-tuning on CASIA-CXR.
    Returns one (image, tokenized_report) pair per index.
    """

    def __init__(
        self,
        csv_path:  str,
        tokenizer,
        split:     str = "train",
    ):
        df = pd.read_csv(csv_path)
        self.df        = preprocess_dataframe(df)
        self.tokenizer = tokenizer
        self.transform = get_transforms(split)
        log.info(f"[{split}] Dataset: {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────
        img = Image.open(row["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        # ── Text ──────────────────────────────────────────────────────────
        encoding = self.tokenizer(
            row["report_text"],
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values":   pixel_values,                              # (3,224,224)
            "input_ids":      encoding["input_ids"].squeeze(0),          # (128,)
            "attention_mask": encoding["attention_mask"].squeeze(0),     # (128,)
            "label":          int(row["label"]),
        }



# %%
class FrenchMedCLIP(nn.Module):
    """
    MedCLIP architecture with BioClinicalBERT replaced by CamemBERT-bio.

    Architecture:
        Image branch:  Swin-B  → linear projection → L2-normalised embed (512-d)
        Text branch:   CamemBERT-bio [CLS] → linear projection → L2-normalised embed (512-d)
        Loss:          Symmetric InfoNCE with learnable temperature τ

    Why we use InfoNCE here (not MedCLIP's semantic matching loss):
        MedCLIP's semantic matching loss requires UMLS entity extraction from English text.
        UMLS French coverage is limited — extracting reliable entities from French clinical
        text is non-trivial and would require a separate NER pipeline (e.g., MedSpaCy-FR).
        For this first French adaptation, symmetric InfoNCE with all_gather (128 negatives
        across 4 GPUs) is the correct and well-validated baseline.
        Future work: adapt the semantic matching loss using DrBERT-based NER.
    """

    def __init__(self):
        super().__init__()

        # ── Vision encoder: Swin-B ─────────────────────────────────────────
        self.vision_encoder = SwinModel.from_pretrained(
            cfg.vision_encoder_name
        )
        vision_hidden_size = self.vision_encoder.config.hidden_size  # 1024 for Swin-B

        # Freeze bottom N stages of Swin
        self._freeze_swin_stages(cfg.freeze_vision_layers)

        # ── Text encoder: CamemBERT-bio ───────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(
            cfg.text_encoder_name
        )

        if hasattr(self.text_encoder, "pooler"):
            self.text_encoder.pooler = None

        text_hidden_size = self.text_encoder.config.hidden_size  # 768

        # Freeze bottom N layers of text encoder
        self._freeze_text_layers(cfg.freeze_text_layers)

        # ── Projection heads (one per branch) ─────────────────────────────
        # Linear projection → cfg.embed_dim (512) — from MedCLIP paper
        self.vision_proj = nn.Linear(vision_hidden_size, cfg.embed_dim)
        self.text_proj   = nn.Linear(text_hidden_size,   cfg.embed_dim)

        # ── Learnable temperature τ ────────────────────────────────────────
        # Initialised at log(1/0.07) ≈ 2.659 — from MedCLIP / CLIP paper
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1 / cfg.temperature)
        )

    def _freeze_swin_stages(self, n_stages: int):
        """Freeze the patch embedding and first n_stages of Swin."""
        # Always freeze patch embedding
        for p in self.vision_encoder.embeddings.parameters():
            p.requires_grad = False
        # Freeze the first n_stages encoder layers
        for i, layer in enumerate(self.vision_encoder.encoder.layers):
            if i < n_stages:
                for p in layer.parameters():
                    p.requires_grad = False
        log.info(f"Swin: frozen patch_embed + first {n_stages} stages")

    def _freeze_text_layers(self, n_layers: int):
        """Freeze embeddings and first n_layers transformer layers."""
        for p in self.text_encoder.embeddings.parameters():
            p.requires_grad = False
        # CamemBERT-bio / DrBERT encoder layers
        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < n_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        log.info(f"Text encoder: frozen embeddings + first {n_layers} layers")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Pass images through Swin, average-pool the patch tokens,
        project to embed_dim, and L2-normalise.
        """
        outputs = self.vision_encoder(pixel_values=pixel_values)
        # outputs.last_hidden_state: (B, num_patches, hidden_size)
        # Mean-pool across patch dimension
        pooled = outputs.last_hidden_state.mean(dim=1)   # (B, hidden_size)
        projected = self.vision_proj(pooled)              # (B, embed_dim)
        return F.normalize(projected, dim=-1)             # L2 normalise

    def encode_text(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass text through CamemBERT-bio, take [CLS] token,
        project to embed_dim, and L2-normalise.
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] token embedding (first token)
        cls_embed = outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)
        projected = self.text_proj(cls_embed)             # (B, embed_dim)
        return F.normalize(projected, dim=-1)             # L2 normalise

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:

        img_embeds  = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        return {
            "img_embeds":  img_embeds,
            "text_embeds": text_embeds,
        }


# %%

def contrastive_loss(
    img_embeds:  torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: nn.Parameter,
    is_distributed: bool = True,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss.

    In DDP training, each GPU only sees batch_size samples (32).
    Without all_gather, each sample only has 31 negatives.
    With all_gather across 4 GPUs: 128 negatives per sample — 4x more signal.

    all_gather collects embeddings from all GPUs before computing loss.
    Gradients still flow through the local GPU's embeddings only
    (gathered tensors are detached), which is standard CLIP practice.
    """
    # Clamp temperature to prevent numerical instability
    # MedCLIP clamps logit_scale to [0, 4.6052] = log(100)
    logit_scale_val = torch.clamp(logit_scale, 0, math.log(100)).exp()

    if is_distributed and dist.is_initialized():
        # Gather embeddings from all GPUs
        world_size = dist.get_world_size()
        rank       = dist.get_rank()

        # Allocate buffers on each GPU
        all_img   = [torch.zeros_like(img_embeds)  for _ in range(world_size)]
        all_text  = [torch.zeros_like(text_embeds) for _ in range(world_size)]

        # All-gather (non-blocking for efficiency)
        dist.all_gather(all_img,  img_embeds)
        dist.all_gather(all_text, text_embeds)

        # Replace local rank's slice with the version that has gradients
        all_img[rank]  = img_embeds
        all_text[rank] = text_embeds

        # Concatenate into full-batch tensors (B_total = 32 × 4 = 128)
        all_img  = torch.cat(all_img,  dim=0)   # (128, embed_dim)
        all_text = torch.cat(all_text, dim=0)   # (128, embed_dim)
    else:
        all_img  = img_embeds
        all_text = text_embeds

    # Compute similarity matrix
    # logits[i,j] = similarity between image i and text j, scaled by τ
    logits_per_image = logit_scale_val * all_img  @ all_text.T   # (N, N)
    logits_per_text  = logits_per_image.T                         # (N, N)

    # Ground truth: diagonal (each image matches its own report)
    N      = all_img.shape[0]
    labels = torch.arange(N, device=all_img.device)

    # Symmetric cross-entropy
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text,  labels)
    loss     = (loss_i2t + loss_t2i) / 2

    return loss



# %%

@torch.no_grad()
def compute_retrieval_metrics(
    model:      FrenchMedCLIP,
    dataloader: DataLoader,
    device:     torch.device,
    k_values:   list = [1, 5, 10],
) -> dict:
    """
    Computes validation loss + Image→Text and Text→Image Recall@k.

    Val loss uses the same symmetric InfoNCE as training but:
      - no all_gather (single GPU inference)
      - no gradient computation
      - temperature is clamped identically to training

    Recall@k: fraction of queries where the correct match
    appears in the top-k retrieved results.
      i2t = given an image, retrieve its report
      t2i = given a report, retrieve its image
    """
    model.eval()
    all_img_embeds  = []
    all_text_embeds = []
    total_val_loss  = 0.0
    n_batches       = 0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast():
            img_embeds  = model.encode_image(pixel_values)
            text_embeds = model.encode_text(input_ids, attention_mask)

            # Val loss — no DDP gather, batch is local only
            loss = contrastive_loss(
                img_embeds=img_embeds,
                text_embeds=text_embeds,
                logit_scale=model.logit_scale,
                is_distributed=False,
            )

        total_val_loss += loss.item()
        n_batches      += 1

        all_img_embeds.append(img_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

    img_embeds  = torch.cat(all_img_embeds,  dim=0)   # (N, embed_dim)
    text_embeds = torch.cat(all_text_embeds, dim=0)   # (N, embed_dim)

    sim    = img_embeds @ text_embeds.T                # (N, N)
    N      = sim.shape[0]
    labels = torch.arange(N)

    metrics = {"val_loss": total_val_loss / max(n_batches, 1)}

    for k in k_values:
        topk_i2t = sim.topk(k, dim=1).indices
        correct  = (topk_i2t == labels.unsqueeze(1)).any(dim=1).float()
        metrics[f"i2t_R@{k}"] = correct.mean().item()

        topk_t2i = sim.T.topk(k, dim=1).indices
        correct  = (topk_t2i == labels.unsqueeze(1)).any(dim=1).float()
        metrics[f"t2i_R@{k}"] = correct.mean().item()

    model.train()
    return metrics


# %%

def train_one_epoch(
    model:          DDP,
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    scaler:         GradScaler,
    scheduler,
    device:         torch.device,
    epoch:          int,
    is_distributed: bool,
    rank:           int,
) -> float:
    """Runs one full training epoch. Returns average loss."""

    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1} [train]",
        disable=(rank != 0),   # Only show progress bar on rank 0
    )

    for batch in pbar:
        pixel_values   = batch["pixel_values"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── Forward pass in fp16 ──────────────────────────────────────────
        with autocast():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = contrastive_loss(
                img_embeds=outputs["img_embeds"],
                text_embeds=outputs["text_embeds"],
                logit_scale=model.module.logit_scale
                    if is_distributed else model.logit_scale,
                is_distributed=is_distributed,
            )

        # ── Backward pass in fp16 ─────────────────────────────────────────
        scaler.scale(loss).backward()

        # Gradient clipping — prevents exploding gradients with small dataset
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                "τ":    f"{(model.module.logit_scale if is_distributed else model.logit_scale).exp().item():.3f}",
            })

    return total_loss / max(n_batches, 1)



# %%
def main():
    # ── Args ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",  action="store_true",
                        help="Single GPU debug mode — no DDP")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── DDP setup ─────────────────────────────────────────────────────────────
    is_distributed = not args.debug and "LOCAL_RANK" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        log.info(f"Training on {world_size} GPU(s)")
        log.info(f"Text encoder:   {cfg.text_encoder_name}")
        log.info(f"Vision encoder: {cfg.vision_encoder_name}")
        log.info(f"Effective batch size: {cfg.batch_size * world_size}")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_name)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = CASIACXRDataset(cfg.train_csv, tokenizer, split="train")
    val_ds   = CASIACXRDataset(cfg.val_csv,   tokenizer, split="val")

    train_sampler = DistributedSampler(train_ds, shuffle=True) \
                    if is_distributed else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) \
                    if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,        # Needed for contrastive loss (consistent batch size)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,   # Larger batch OK for inference
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = FrenchMedCLIP().to(device)

    if is_distributed:
        # SyncBatchNorm converts any BN layers to sync across GPUs
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate LR for projection heads (higher) vs encoder layers (lower)
    # This prevents the projection heads from lagging behind the encoders
    encoder_params    = []
    projection_params = []

    base_model = model.module if is_distributed else model
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if "proj" in name or "logit_scale" in name:
            projection_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,    "lr": cfg.lr},
        {"params": projection_params, "lr": cfg.lr * 5},  # Higher LR for new layers
    ], weight_decay=cfg.weight_decay)

    # ── Scheduler: warmup + cosine decay ─────────────────────────────────────
    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / max(warmup_steps, 1)
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── fp16 scaler ───────────────────────────────────────────────────────────
    scaler = GradScaler()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_recall = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", 0.0)
        if rank == 0:
            log.info(f"Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, cfg.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)  # Ensures different shuffle each epoch

        # Train
        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            is_distributed=is_distributed,
            rank=rank,
        )

        # Validate — only on rank 0 (or single GPU)
        if rank == 0:
            eval_model = model.module if is_distributed else model
            metrics = compute_retrieval_metrics(
                model=eval_model,
                dataloader=val_loader,
                device=device,
            )

            # Use mean of i2t and t2i Recall@10 as the primary metric
            recall_10 = (metrics["i2t_R@10"] + metrics["t2i_R@10"]) / 2

            log.info(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"Loss: {avg_loss:.4f} | Val loss: {metrics['val_loss']:.4f} | "
                f"i2t R@1: {metrics['i2t_R@1']:.3f} | "
                f"i2t R@10: {metrics['i2t_R@10']:.3f} | "
                f"t2i R@1: {metrics['t2i_R@1']:.3f} | "
                f"t2i R@10: {metrics['t2i_R@10']:.3f}"
            )

            history.append({
                "epoch":         epoch + 1,
                "train_loss":    avg_loss,           # renamed from "loss" for clarity
                "val_loss":      metrics["val_loss"],
                "i2t_R@1":       metrics["i2t_R@1"],
                "i2t_R@5":       metrics["i2t_R@5"],
                "i2t_R@10":      metrics["i2t_R@10"],
                "t2i_R@1":       metrics["t2i_R@1"],
                "t2i_R@5":       metrics["t2i_R@5"],
                "t2i_R@10":      metrics["t2i_R@10"],
                "lr":            scheduler.get_last_lr()[0],
                "temperature":   base_model.logit_scale.exp().item(),
                "timestamp":     datetime.now().isoformat(),
            })

            # Save best checkpoint
            if recall_10 > best_recall:
                best_recall = recall_10
                torch.save(
                    {
                        "epoch":        epoch,
                        "model":        eval_model.state_dict(),
                        "optimizer":    optimizer.state_dict(),
                        "scheduler":    scheduler.state_dict(),
                        "scaler":       scaler.state_dict(),
                        "best_recall":  best_recall,
                        "metrics":      metrics,
                        "config": {
                            "text_encoder":   cfg.text_encoder_name,
                            "vision_encoder": cfg.vision_encoder_name,
                            "embed_dim":      cfg.embed_dim,
                        },
                    },
                    Path(cfg.output_dir) / "best_model.pt",
                )
                log.info(f"  ✓ New best model saved (Recall@10={best_recall:.3f})")

            # Save latest checkpoint (for resuming)
            torch.save(
                {
                    "epoch":       epoch,
                    "model":       eval_model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "scaler":      scaler.state_dict(),
                    "best_recall": best_recall,
                },
                Path(cfg.output_dir) / "latest.pt",
            )

            # Save training history
            with open(Path(cfg.output_dir) / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    if rank == 0:
        log.info(f"Training complete. Best Recall@10: {best_recall:.3f}")
        # ── Results summary ───────────────────────────────────────────────
        best_epoch_entry = max(history, key=lambda x: (x["i2t_R@10"] + x["t2i_R@10"]) / 2)
        summary = {
            "run_date":           datetime.now().isoformat(),
            "text_encoder":       cfg.text_encoder_name,
            "vision_encoder":     cfg.vision_encoder_name,
            "epochs_trained":     epoch + 1,
            "best_epoch":         best_epoch_entry["epoch"],
            "best_recall10_mean": (best_epoch_entry["i2t_R@10"] + best_epoch_entry["t2i_R@10"]) / 2,
            "best_metrics":       best_epoch_entry,
            "config": {
                "batch_size_per_gpu":   cfg.batch_size,
                "effective_batch_size": cfg.batch_size * world_size,
                "lr":                   cfg.lr,
                "max_length":           cfg.max_length,
                "embed_dim":            cfg.embed_dim,
                "freeze_text_layers":   cfg.freeze_text_layers,
                "freeze_vision_layers": cfg.freeze_vision_layers,
                "text_sections":        cfg.text_sections,
            },
        }
        with open(Path(cfg.output_dir) / "results_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results summary saved → {cfg.output_dir}/results_summary.json")

    if is_distributed:
        dist.destroy_process_group()


# %%


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()

# ══════════════════════════════════════════════════════════════════════════════
# HOW TO RUN
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#    pip install torch torchvision transformers pandas Pillow opencv-python tqdm
#    pip install sentencepiece  # required for CamemBERT tokenizer
#
# 2. Prepare CSVs (from your preprocessing step):
#    data/train.csv, data/val.csv, data/test.csv
#
# 3. Debug on single GPU first:
#    python train.py --debug
#
# 4. Full training on 4 GPUs:
#    torchrun --nproc_per_node=4 train.py
#
# 5. Resume from checkpoint:
#    torchrun --nproc_per_node=4 train.py --resume checkpoints/latest.pt
#
# Expected memory usage per 2080 Ti (11GB):
#    Swin-B:         ~1.5 GB
#    CamemBERT-bio:  ~0.5 GB
#    Batch (32×224): ~1.0 GB
#    Gradients+Adam: ~4.0 GB
#    Total:          ~7–8 GB  → fits in 11GB with headroom
#
# Expected training time on 4×2080 Ti:
#    ~8,800 train samples / 32 batch / 4 GPUs = ~70 batches/epoch
#    ~2–3 min/epoch → 30 epochs ≈ 1–1.5 hours total



