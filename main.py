import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from PIL import Image

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class LoRALinear(nn.Module):
    def __init__(self, original, r=2, dropout=0.25):
        super().__init__()
        self.original = original
        self.r = r
        self.A = nn.Parameter(torch.randn(original.out_features, r))
        self.B = nn.Parameter(torch.randn(r, original.in_features))
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (r ** 0.5)

    def forward(self, x):
        return self.original(x) + self.scale * self.dropout((x @ self.B.T) @ self.A.T)


def apply_lora_to_vit(model, r=2, dropout=0.25, blocks=12):
    for block in model.blocks[-blocks:]:
        attn = block.attn
        attn.qkv = LoRALinear(attn.qkv, r=r, dropout=dropout)
        attn.proj = LoRALinear(attn.proj, r=r, dropout=dropout)
    return model


class FAMSeC(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.model = timm.create_model('vit_xsmall_patch16_clip_224', pretrained=True)
        self.model = apply_lora_to_vit(self.model, r=r)

    def encode_image(self, x):
        features = self.model.forward_features(x)
        return features[:, 0]  # Extract CLS token


def get_dataloaders(data_path, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    print(f"Loaded {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def contrastive_loss(Eg, Et, labels, tau=0.07):
    sim = torch.mm(F.normalize(Eg), F.normalize(Et).T) / tau
    li = labels.unsqueeze(1) == labels.unsqueeze(0)
    li = li.float()
    loss = - (li * F.logsigmoid(sim) + (1 - li) * F.logsigmoid(-sim)).mean()
    return loss

def train(model, guide_model, dataloader, optimizer, device):
    model.train()
    guide_model.eval()
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            Eg = guide_model.forward_features(images)[:, 0]  # CLS token
        Et = model.encode_image(images)
        loss = contrastive_loss(Eg, Et, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(model, test_img, real_feats, fake_feats):
    model.eval()
    with torch.no_grad():
        embed = model.encode_image(test_img)
        dr = F.cosine_similarity(embed, real_feats).mean()
        df = F.cosine_similarity(embed, fake_feats).mean()
    return "REAL" if dr >= df else "FAKE"

# ----------------------------
# Utility: Load Test Image
# ----------------------------
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)

# ----------------------------
# Evaluate on Test Set
# ----------------------------
def evaluate(model, real_feats, fake_feats, device):
    correct = 0
    total = 0

    for cls in ["REAL", "FAKE"]:
        folder = f"./dataset/test/{cls}"
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = load_image(os.path.join(folder, fname)).to(device)
                pred = predict(model, img, real_feats, fake_feats)
                print("correct: "+cls)
                print("pred: "+pred)
                correct += int(pred == cls)
                total += 1

    acc = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")

# ----------------------------
# Main Entrypoint
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    trainable_model = FAMSeC().to(device)
    guide_model = timm.create_model("vit_xsmall_patch16_clip_224", pretrained=True).to(device)
    guide_model.eval()

    # Dataset
    dataloader = get_dataloaders("./dataset/train/", batch_size=32)

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_model.parameters(), lr=1e-4)

    # Train
    train(trainable_model, guide_model, dataloader, optimizer, device)

    # Generate embeddings for prediction
    real_imgs, _ = next(iter(dataloader))
    real_feats = trainable_model.encode_image(real_imgs.to(device)[:10])
    fake_feats = trainable_model.encode_image(real_imgs.to(device)[-10:])

    # Evaluate
    evaluate(trainable_model, real_feats, fake_feats, device)

if __name__ == "__main__":
    main()
