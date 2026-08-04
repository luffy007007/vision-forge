

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT, ImageFolder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/eurosat/2750"   # class-named subfolders, created by the fetch below
MODEL_NAME = "convnext_tiny"
BATCH = 64


tmp = timm.create_model(MODEL_NAME, pretrained=False)
cfg = timm.data.resolve_data_config({}, model=tmp)
base_tf = timm.data.create_transform(**cfg)


train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    base_tf,
])

EuroSAT("data", download=True)

full = ImageFolder(DATA_DIR, transform=base_tf)
CLASSES = full.classes
NUM_CLASSES = len(CLASSES)   # read from the folder, never hardcoded
print(f"{NUM_CLASSES} classes:", CLASSES)

# two views of the same folder
augmented = ImageFolder(DATA_DIR, transform=train_tf)

n_val = int(0.2 * len(full))
perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(0)).tolist()
train_ds = Subset(augmented, perm[n_val:])
val_ds = Subset(full, perm[:n_val])

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_epochs(n, lr):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(n):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()              # gradients accumulate by default — clear them
            loss = criterion(model(x), y)
            loss.backward()              # compute gradients
            opt.step()                   # update weights
        print(f"  epoch {epoch+1}/{n}  val_acc={evaluate():.4f}")


# Run A: freeze everything except the classifier head
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True
print("RUN A (frozen backbone, head only):")
train_epochs(5, lr=1e-3)
acc_a = evaluate()

#Run B: unfreeze the last block, fine-tune gently

for p in model.get_submodule("stages.3").parameters():   # convnext_tiny has 4 stages
    p.requires_grad = True
print("RUN B (last block unfrozen, lr/10):")
train_epochs(5, lr=1e-4)
acc_b = evaluate()

print(f"\nFrozen head:   {acc_a:.4f}")
print(f"Fine-tuned:    {acc_b:.4f}")
print("^ put both numbers in your README. The delta is the story.")


torch.save({"model": MODEL_NAME,
            "classes": CLASSES,
            "state_dict": model.state_dict()},
           "classifier.pt")  
## downloaded in models folder