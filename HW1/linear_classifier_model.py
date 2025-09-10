# 0) Setup
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import random

# Reproducibility
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1) Data: load MNIST and filter to labels {3, 8}
transform = transforms.Compose([
    transforms.ToTensor(),                       # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))   # standard MNIST normalization
])

train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_full  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def indices_for_labels(ds, keep={3,8}):
    return [i for i,(_,y) in enumerate(ds) if y in keep]

train_idx = indices_for_labels(train_full, {3,8})
test_idx  = indices_for_labels(test_full,  {3,8})

train_ds = Subset(train_full, train_idx)
test_ds  = Subset(test_full,  test_idx)

# Map labels: 3 -> 0, 8 -> 1 (binary)
def relabel(batch):
    x, y = batch
    y = (y == 8).long()
    return x, y

# Simple wrapper to relabel inside a DataLoader
class RelabeledLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            yield relabel(batch)

batch_size = 128
train_loader = RelabeledLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = RelabeledLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 2) Model: a single linear layer (logistic regression)
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 1)  # 784 -> 1 logit

    def forward(self, x):
        x = x.view(x.size(0), -1)      # flatten
        logit = self.fc(x)
        return logit.squeeze(1)        # [B]

model = LinearClassifier().to(device)

# 3) Loss & optimizer
# BCEWithLogitsLoss expects float targets in {0,1}
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# (You can also try Adam(lr=1e-3) if you prefer)

# 4) Training loop
def train_epoch():
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()      # P(class=8)
        all_p.append(probs)
        all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    p = np.concatenate(all_p)
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)  # rows=true, cols=pred
    return acc, cm, y_true, y_pred, p

epochs = 10
for ep in range(1, epochs+1):
    tr_loss = train_epoch()
    acc, cm, *_ = evaluate(test_loader)
    print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | test acc {acc:.4f}")
print("Confusion matrix (rows=true [3->0, 8->1], cols=pred):\n", cm)

# 5) Optional: a quick report
@torch.no_grad()
def report(loader):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        yhat = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        ys.append(y.numpy()); preds.append(yhat)
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    print(classification_report(ys, preds, target_names=["3","8"]))

report(test_loader)

# 6) Save the model (weights + class mapping note)
torch.save({"state_dict": model.state_dict(),
            "normalize_mean": 0.1307, "normalize_std": 0.3081,
            "label_map": {"3":0, "8":1}}, "mnist_3v8_linear.pt")
print("Saved to mnist_3v8_linear.pt")
