"""
clip_lora_train_dual_contrastive.py

Full CLIP + LoRA training with TWO CLIP models:
- A frozen reference CLIP (no LoRA, eval mode) -> provides stable targets
- A trainable CLIP with LoRA adapters -> updated to align with reference via contrastive loss

Training objective:
  total_loss = alpha_ce * CE( logits(image_lora, class_text) )
              + beta_contrastive * InfoNCE( image_lora || image_ref )

Where:
- CE uses precomputed class text features from the FROZEN model (kept fixed during training)
- InfoNCE encourages the LoRA image features to be close to the frozen model's image features
  for the SAME image and far from those of other images in the batch.

The original ImageFolder-like dataset layout is preserved:
  root_dir/
    train/{fake,real}
    test/{fake,real}

If you hit errors related to "target_modules" or module names, paste the error and I'll adjust.
"""

import os
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel, logging as hf_logging
from peft import get_peft_model, LoraConfig, TaskType

hf_logging.set_verbosity_error()  # reduce HF logs


# ---------------------------
# Dataset - simple ImageFolder-like
# ---------------------------
class SimpleImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, class_names: List[str], templates: List[str], processor: CLIPProcessor):
        self.samples = []
        self.class_names = class_names
        self.templates = templates
        self.processor = processor

        for label, cname in enumerate(class_names):
            folder = os.path.join(root_dir, cname)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected folder not found: {folder}")
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, cname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cname, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        pixel_inputs = {k: v.squeeze(0) for k, v in pixel_inputs.items()}
        return pixel_inputs, cname, torch.tensor(label, dtype=torch.long), os.path.basename(path)


def collate_fn(batch):
    """
    Batch is a list of tuples: (pixel_inputs_dict, classname, label, filename)
    We need to stack pixel_values into (B, C, H, W), gather labels, classnames, filenames.
    """
    pixel_values = torch.stack([item[0]["pixel_values"] for item in batch], dim=0)
    labels = torch.stack([item[2] for item in batch], dim=0)
    classnames = [item[1] for item in batch]
    filenames = [item[3] for item in batch]
    return {"pixel_values": pixel_values}, classnames, labels, filenames


# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def compute_text_features_for_classes(clip_model: CLIPModel, processor: CLIPProcessor,
                                     classnames: List[str], templates: List[str], device: torch.device) -> torch.Tensor:
    """
    Returns normalized text_features of shape (num_classes, dim). Uses the provided clip_model.
    """
    clip_model.eval()
    text_features = []
    for cname in classnames:
        prompts = [t.format(cname.replace('_', ' ')) for t in templates]
        enc = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        embeddings = clip_model.get_text_features(**enc)  # (num_templates, dim)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean_emb = embeddings.mean(dim=0)
        mean_emb = mean_emb / mean_emb.norm()
        text_features.append(mean_emb)
    text_features = torch.stack(text_features, dim=0)  # (num_classes, dim)
    return text_features


def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Standard (non-symmetric) InfoNCE:
      - anchors: (B, D) -> here from LoRA model
      - positives: (B, D) -> here from frozen model (same images, aligned by index)
      - negatives are the other rows in the batch
    """
    anchors = anchors / anchors.norm(dim=-1, keepdim=True)
    positives = positives / positives.norm(dim=-1, keepdim=True)
    logits = anchors @ positives.t()  # (B, B)
    logits = logits / max(1e-8, temperature)
    targets = torch.arange(anchors.size(0), device=anchors.device)
    return nn.CrossEntropyLoss()(logits, targets)


@torch.no_grad()
def evaluate(model: CLIPModel, dataloader: DataLoader, text_features: torch.Tensor,
             classnames: List[str], device: torch.device, logit_scale_value: float) -> Tuple[float, list, list]:
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        pixel_inputs, _batch_classnames, labels, files = batch
        pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
        labels = labels.to(device)
        image_features = model.get_image_features(**pixel_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.t()) * logit_scale_value
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, all_preds, all_labels


# ---------------------------
# Train wrapper (two-model setup)
# ---------------------------

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    print("Loading CLIP processor and TWO models (frozen + LoRA trainable)...")
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    # 1) Frozen reference model (no LoRA) -------------------------------------
    ref_model = CLIPModel.from_pretrained(args.clip_model_name)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    ref_model.to(device)

    # 2) Trainable model (with LoRA) ------------------------------------------
    train_model = CLIPModel.from_pretrained(args.clip_model_name)
    if args.apply_lora:
        target_modules = args.target_modules.split(",") if isinstance(args.target_modules, str) else args.target_modules
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        print(f"Applying LoRA to modules matching: {target_modules}")
        train_model = get_peft_model(train_model, lora_config)
    else:
        print("WARNING: --apply_lora not set. Training full model parameters (not recommended).")
    train_model.to(device)

    # Datasets / Loaders ------------------------------------------------------
    classnames = args.class_names
    templates = args.templates

    print("Preparing datasets...")
    train_root = os.path.join(args.root_dir, "train")
    test_root = os.path.join(args.root_dir, "test")

    train_ds = SimpleImageFolderDataset(train_root, classnames, templates, processor)
    test_ds = SimpleImageFolderDataset(test_root, classnames, templates, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=args.drop_last)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    # Precompute CLASS TEXT features using the FROZEN reference model ---------
    print("Computing class text features with the frozen reference model...")
    text_features = compute_text_features_for_classes(ref_model, processor, classnames, templates, device)
    text_features = text_features.to(device)

    # Temperatures / logit scales --------------------------------------------
    try:
        ref_logit_scale = ref_model.logit_scale.exp().item()
    except Exception:
        ref_logit_scale = 1.0
    try:
        train_logit_scale = train_model.logit_scale.exp().item()
    except Exception:
        train_logit_scale = 1.0

    # Optimizer only for trainable params (PEFT usually makes only the adapter params trainable)
    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Optional LR scheduler (per-step cosine)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * steps_per_epoch), eta_min=1e-7
    )

    ce_criterion = nn.CrossEntropyLoss()

    # --------------------------- Training Loop ---------------------------
    for epoch in range(args.epochs):
        train_model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for pixel_inputs, batch_classnames, labels, filenames in pbar:
            pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()

            # 1) Forward LoRA model (trainable)
            image_feats_lora = train_model.get_image_features(**pixel_inputs)  # (B, D)
            image_feats_lora = image_feats_lora / image_feats_lora.norm(dim=-1, keepdim=True)

            # 2) Forward FROZEN model (no-grad already)
            with torch.no_grad():
                image_feats_ref = ref_model.get_image_features(**pixel_inputs)  # (B, D)
                image_feats_ref = image_feats_ref / image_feats_ref.norm(dim=-1, keepdim=True)

            # 3) CE loss (classification) using class text features (from frozen model)
            logits_ce = (image_feats_lora @ text_features.t()) * train_logit_scale
            ce_loss = ce_criterion(logits_ce, labels)

            # 4) Contrastive InfoNCE between LoRA features (anchors) and Ref features (positives)
            contrastive_temp = args.contrastive_temperature if args.contrastive_temperature > 0 else (1.0 / max(1e-8, train_logit_scale))
            nce_loss = info_nce_loss(image_feats_lora, image_feats_ref, temperature=contrastive_temp)

            # 5) Combine losses
            loss = args.alpha_ce * ce_loss + args.beta_contrastive * nce_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            bsz = labels.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
            avg_loss = total_loss / max(1, total_samples)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ce": f"{ce_loss.item():.3f}", "nce": f"{nce_loss.item():.3f}"})

        # End epoch: evaluate CURRENT (LoRA) model on test set
        acc, _, _ = evaluate(train_model, test_loader, text_features, classnames, device, train_logit_scale)
        print(f"Epoch {epoch+1}/{args.epochs} finished. Train loss: {avg_loss:.4f}  Test Acc (LoRA): {acc:.2f}%")

        # Save epoch checkpoint of PEFT adapters (if any)
        if args.apply_lora and args.save_prefix:
            peft_save_dir = f"{args.save_prefix}_epoch{epoch+1}"
            print(f"Saving PEFT adapters to {peft_save_dir} ...")
            train_model.save_pretrained(peft_save_dir)

    # --------------------------- Final Evaluation ---------------------------
    final_acc, preds, labels_list = evaluate(train_model, test_loader, text_features, classnames, device, train_logit_scale)
    print(f"\nFinal Test Accuracy (LoRA): {final_acc:.2f}%")

    # Show some random sample predictions
    print("\nSample predictions (first 10, stratified):")
    with torch.no_grad():
        real_samples = []
        fake_samples = []
        for pixel_inputs, batch_classnames, labels, filenames in test_loader:
            pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
            labels = labels.to(device)

            image_features = train_model.get_image_features(**pixel_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.t()) * train_logit_scale
            preds = logits.argmax(dim=1).cpu().tolist()

            for fname, p, gt in zip(filenames, preds, labels.cpu().tolist()):
                if classnames[gt].lower() == "real" and len(real_samples) < 5:
                    real_samples.append((fname, classnames[p], classnames[gt]))
                elif classnames[gt].lower() == "fake" and len(fake_samples) < 5:
                    fake_samples.append((fname, classnames[p], classnames[gt]))

            if len(real_samples) >= 5 and len(fake_samples) >= 5:
                break

        random.shuffle(real_samples)
        random.shuffle(fake_samples)

        print("Random REAL samples:")
        for fname, pred, gt in real_samples:
            print(f"{fname} -> Pred: {pred}  GT: {gt}")

        print("\nRandom FAKE samples:")
        for fname, pred, gt in fake_samples:
            print(f"{fname} -> Pred: {pred}  GT: {gt}")

    if args.apply_lora and args.save_prefix:
        print(f"Saving final PEFT adapters to {args.save_prefix} ...")
        train_model.save_pretrained(args.save_prefix)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="dataset",
                        help="Root dataset folder containing train/ and test/ subfolders")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32",
                        help="Hugging Face CLIP model name")
    parser.add_argument("--class_names", nargs="+", default=["fake", "real"])
    parser.add_argument("--templates", nargs="+", default=["a photo of a {}"])

    # LoRA / PEFT
    parser.add_argument("--apply_lora", action="store_true", help="Apply LoRA adapters via PEFT to the trainable model")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated module name substrings for LoRA (e.g., 'q_proj,v_proj')")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--drop_last", action="store_true", help="Drop last incomplete batch (useful for InfoNCE)")

    # Loss mixing
    parser.add_argument("--alpha_ce", type=float, default=1.0, help="Weight for CE classification loss")
    parser.add_argument("--beta_contrastive", type=float, default=1.0, help="Weight for InfoNCE contrastive loss")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE (if <=0 uses 1/logit_scale)")

    # Saving
    parser.add_argument("--save_prefix", type=str, default="clip_peft_adapters",
                        help="If set and apply_lora=True, save PEFT adapters to this prefix directory")
    parser.add_argument("--force_cpu", action="store_true", help="Don't use GPU even if available")

    args = parser.parse_args()

    run_training(args)
