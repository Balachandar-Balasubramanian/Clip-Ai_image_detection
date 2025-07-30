import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel
from peft import get_peft_model, LoraConfig, TaskType

class ForenSynthDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        for label, cls in enumerate(['REAL', 'FAKE']):
            folder = os.path.join(root, cls)
            for img_name in os.listdir(folder):
                self.paths.append(os.path.join(folder, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class CLIPWithFAM(nn.Module):
    def __init__(self, lora_rank=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.25,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.clip = get_peft_model(self.clip, peft_config)

    def forward(self, x):
        return self.clip.get_image_features(pixel_values=x)

def contrastive_loss(Eg, Et, labels, tau=0.07):
    sim_matrix = torch.matmul(Eg, Et.T) / tau  # [N, N]
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N], True for positives

    positives = sim_matrix[label_matrix]
    negatives = sim_matrix[~label_matrix]

    pos_loss = -torch.log(torch.sigmoid(positives)).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(negatives)).mean()
    return pos_loss + neg_loss


def train_famsec():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
    train_dataset = ForenSynthDataset("./dataset/train/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    T_model = CLIPWithFAM().to(device)
    G_model = CLIPWithFAM().eval().to(device)

    optimizer = torch.optim.Adam(T_model.parameters(), lr=1e-4)

    print("Training FAMSeC...")
    for epoch in range(10):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                Eg = G_model(imgs)
            Et = T_model(imgs)
            loss = contrastive_loss(Eg, Et, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(T_model.state_dict(), "famsec.pth")
    print("Model saved to famsec.pth")
    return T_model

def predict():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPWithFAM().to(device)
    model.load_state_dict(torch.load("famsec.pth"))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def embed_folder(folder):
        embs = []
        for fname in os.listdir(folder):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img)
            embs.append(emb)
        return torch.stack(embs).mean(0)

    real_emb = embed_folder('./dataset/test/REAL')
    fake_emb = embed_folder('./dataset/test/FAKE')

    test_image = "./dataset/test/FAKE/sample.jpg"
    img = Image.open(test_image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)

    d_real = F.cosine_similarity(emb, real_emb.unsqueeze(0)).item()
    d_fake = F.cosine_similarity(emb, fake_emb.unsqueeze(0)).item()
    label = "REAL" if d_real > d_fake else "FAKE"
    print(f"Prediction for {test_image}: {label}")


def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPWithFAM().to(device)
    model.load_state_dict(torch.load("famsec.pth"))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def embed_folder(folder):
        embs = []
        for fname in os.listdir(folder):
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img)
            embs.append(emb)
        return torch.stack(embs).mean(0)

    real_emb = embed_folder('./dataset/test/REAL')
    fake_emb = embed_folder('./dataset/test/FAKE')

    print("Evaluating on test dataset...")

    correct = 0
    total = 0

    # Evaluate REAL class
    for fname in os.listdir('./dataset/test/REAL'):
        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join('./dataset/test/REAL', fname)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img)
            d_real = F.cosine_similarity(emb, real_emb.unsqueeze(0)).item()
            d_fake = F.cosine_similarity(emb, fake_emb.unsqueeze(0)).item()
            predicted = "REAL" if d_real > d_fake else "FAKE"
            if predicted == "REAL":
                correct += 1
            total += 1

    # Evaluate FAKE class
    for fname in os.listdir('./dataset/test/FAKE'):
        if fname.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join('./dataset/test/FAKE', fname)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img)
            d_real = F.cosine_similarity(emb, real_emb.unsqueeze(0)).item()
            d_fake = F.cosine_similarity(emb, fake_emb.unsqueeze(0)).item()
            predicted = "REAL" if d_real > d_fake else "FAKE"
            if predicted == "FAKE":
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")

if __name__ == "__main__":
    train_famsec()
    predict()
    evaluate()

