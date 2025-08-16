"""
clip_lora_train.py

Full CLIP + LoRA training script (Hugging Face transformers + PEFT).
- Uses CLIPModel + CLIPProcessor
- Applies LoRA using peft.get_peft_model
- Trains on train/real and train/fake
- Evaluates on test/real and test/fake

If you hit errors related to "target_modules" or module names, paste the error here and I'll fine-tune it for your installed versions.
"""

import os
import argparse
import random
from typing import List

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
                    # We'll keep text prompts per sample (same template applied)
                    self.samples.append((path, cname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cname, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        # We'll return raw path + processed image and the label; text tokenization handled in training loop
        pixel_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        # pixel_inputs: dict with 'pixel_values' shape (1, C, H, W)
        # squeeze to remove batch dim for collate / dataloader
        pixel_inputs = {k: v.squeeze(0) for k, v in pixel_inputs.items()}
        return pixel_inputs, cname, torch.tensor(label, dtype=torch.long), os.path.basename(path)


# ---------------------------
# Utility: collate for DataLoader
# ---------------------------
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
# Training / Eval functions
# ---------------------------
def compute_text_features_for_classes(clip_model: CLIPModel, processor: CLIPProcessor, classnames: List[str],
                                     templates: List[str], device):
    """
    Returns text_features (num_classes, dim) normalized, and text_inputs tokenization if needed later.
    We'll compute each class by tokenizing the templates and averaging.
    """
    clip_model.eval()
    text_features = []
    text_inputs_cache = []  # store tokenized inputs for each class (useful if you want to re-encode)
    with torch.no_grad():
        for cname in classnames:
            prompts = [t.format(cname.replace('_', ' ')) for t in templates]
            enc = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            # rely on CLIPModel.get_text_features which handles text encoder
            embeddings = clip_model.get_text_features(**enc)  # (num_templates, dim)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            mean_emb = embeddings.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            text_features.append(mean_emb)
            text_inputs_cache.append(enc)  # keep the tokenized input for possible later use
    text_features = torch.stack(text_features, dim=0)  # (num_classes, dim)
    return text_features, text_inputs_cache


def train_epoch(clip_model: CLIPModel, processor: CLIPProcessor, dataloader: DataLoader,
                classnames: List[str], templates: List[str], optimizer, device, logit_scale):
    clip_model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Train", leave=False):
        pixel_inputs, batch_classnames, labels, _files = batch
        # pixel_inputs: dict with pixel_values (B, C, H, W)
        pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
        labels = labels.to(device)

        # Build text features for this iteration (we can compute full class text features once and reuse)
        # Here we compute text_features once outside loop normally; but for clarity we'll compute per epoch outside.
        # So this function will expect the caller to have prepared text_features; to keep it simple here we'll assume
        # the caller has computed text_features and passed via closure (we will prepare outside and use global var).
        raise RuntimeError("train_epoch should be called with precomputed text_features; see wrapper train() below.")


def evaluate(clip_model: CLIPModel, processor: CLIPProcessor, dataloader: DataLoader,
             classnames: List[str], text_features, device, logit_scale):
    clip_model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            pixel_inputs, batch_classnames, labels, files = batch
            pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
            labels = labels.to(device)
            # get image features
            image_features = clip_model.get_image_features(**pixel_inputs)  # (B, dim)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features passed in is (num_classes, dim), normalized
            # compute cosine similarity logits (B, num_classes)
            logits = (image_features @ text_features.t()) * logit_scale
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, all_preds, all_labels


# ---------------------------
# Train wrapper that uses precomputed text_features for efficiency
# ---------------------------
def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    # Load model + processor
    print("Loading CLIP model and processor...")
    clip_model = CLIPModel.from_pretrained(args.clip_model_name)
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    # Optionally apply LoRA via PEFT
    if args.apply_lora:
        # This lora_config tries to match common projection names used by CLIP implementations.
        # You might need to tweak target_modules if your version uses different names.
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
        clip_model = get_peft_model(clip_model, lora_config)
    else:
        print("LoRA not applied; full model (or selective heads) will be used.")

    clip_model.to(device)

    # Prepare datasets
    classnames = args.class_names
    templates = args.templates

    print("Preparing datasets...")
    train_root = os.path.join(args.root_dir, "train")
    test_root = os.path.join(args.root_dir, "test")

    train_ds = SimpleImageFolderDataset(train_root, classnames, templates, processor)
    test_ds = SimpleImageFolderDataset(test_root, classnames, templates, processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    # Precompute text features for classes once (on device)
    print("Computing text features for class prompts...")
    text_features, text_inputs_cache = compute_text_features_for_classes(clip_model, processor, classnames, templates,
                                                                         device)
    # text_features: (num_classes, dim)
    # We'll normalize and keep it on device
    text_features = text_features.to(device)
    # logit_scale param in HF CLIPModel: it is stored as logit_scale (learnable)
    try:
        logit_scale = clip_model.logit_scale.exp().item()
    except Exception:
        # fallback to 1.0
        logit_scale = 1.0

    # Set up optimizer only for trainable parameters (PEFT makes only LoRA params trainable typically)
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # scheduler optional
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)),
                                                           eta_min=1e-7)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        clip_model.train()
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for pixel_inputs, batch_classnames, labels, filenames in pbar:
            pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
            labels = labels.to(device)

            # Forward
            optimizer.zero_grad()
            # image features
            image_features = clip_model.get_image_features(**pixel_inputs)  # (B, dim)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # compute logits
            logits = (image_features @ text_features.t()) * logit_scale  # (B, num_classes)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # scheduler step optionally per batch
            if hasattr(scheduler, 'step'):
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            avg_loss = total_loss / max(1, total_samples)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # End epoch: evaluate
        acc, _, _ = evaluate(clip_model, processor, test_loader, classnames, text_features, device, logit_scale)
        print(f"Epoch {epoch+1}/{args.epochs} finished. Train loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")

        # optional: save checkpoint of PEFT adapters only
        if args.apply_lora and args.save_prefix:
            peft_save_dir = f"{args.save_prefix}_epoch{epoch+1}"
            print(f"Saving PEFT adapters to {peft_save_dir} ...")
            clip_model.save_pretrained(peft_save_dir)

    # Final evaluation
    final_acc, preds, labels = evaluate(clip_model, processor, test_loader, classnames, text_features, device, logit_scale)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

    # show some sample predictions
    print("\nSample predictions (first 10):")
    with torch.no_grad():
        real_samples = []
        fake_samples = []

        for pixel_inputs, batch_classnames, labels, filenames in test_loader:
            pixel_inputs = {k: v.to(device) for k, v in pixel_inputs.items()}
            labels = labels.to(device)
            
            image_features = clip_model.get_image_features(**pixel_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (image_features @ text_features.t()) * logit_scale
            preds = logits.argmax(dim=1).cpu().tolist()

            for fname, p, gt in zip(filenames, preds, labels.cpu().tolist()):
                if classnames[gt].lower() == "real" and len(real_samples) < 5:
                    real_samples.append((fname, classnames[p], classnames[gt]))
                elif classnames[gt].lower() == "fake" and len(fake_samples) < 5:
                    fake_samples.append((fname, classnames[p], classnames[gt]))

            # Stop when we have enough samples
            if len(real_samples) >= 5 and len(fake_samples) >= 5:
                break

        # Randomize order within each group
        random.shuffle(real_samples)
        random.shuffle(fake_samples)

        print("Random REAL samples:")
        for fname, pred, gt in real_samples:
            print(f"{fname} -> Pred: {pred}  GT: {gt}")

        print("\nRandom FAKE samples:")
        for fname, pred, gt in fake_samples:
            print(f"{fname} -> Pred: {pred}  GT: {gt}")

    # final save if requested
    if args.apply_lora and args.save_prefix:
        print(f"Saving final PEFT adapters to {args.save_prefix} ...")
        clip_model.save_pretrained(args.save_prefix)


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
    parser.add_argument("--apply_lora", action="store_true", help="Apply LoRA adapters via PEFT")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated target module names (substring match) for LoRA (try 'q_proj,v_proj' first)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_prefix", type=str, default="clip_peft_adapters",
                        help="If set and apply_lora=True, save PEFT adapters to this prefix directory")
    parser.add_argument("--force_cpu", action="store_true", help="Don't use GPU even if available")
    args = parser.parse_args()

    run_training(args)
