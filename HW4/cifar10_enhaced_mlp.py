#!/usr/bin/env python3
"""
Residual Deep MLP for CIFAR-10 (PyTorch, standalone)
- Flattens 32x32x3 images to 3072
- Pixel normalization to [0,1] via ToTensor()
- Architecture: input projection -> (L/2) residual blocks (each 2x Dense with pre-LayerNorm) -> classifier
- Use CrossEntropyLoss (softmax implicit) + Adam optimizer
- Logs per-epoch metrics and saves curves and best checkpoint

Run:
    python cifar10_residual_mlp_pytorch.py --epochs 80 --layers 10 --hidden 512 --batch-size 256 --lr 1e-3
"""
import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------- Args & Seed ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=80, help='Training epochs (50-100 suggested)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden', type=int, default=512, help='Hidden layer width')
    p.add_argument('--layers', type=int, default=10, help='Total number of hidden layers (must be even for residual blocks)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--outdir', type=str, default='outputs_cifar10_resmlp')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------- Residual MLP (pre-LN) ----------------------

class PreLNLinear(nn.Module):
    """LayerNorm -> Linear -> ReLU."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.linear(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    """Two Dense layers with pre-LN and identity skip: y = x + f(x)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layer1 = PreLNLinear(hidden_dim)
        self.layer2 = PreLNLinear(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        return residual + x

class DeepResMLP(nn.Module):
    def __init__(self, input_dim: int = 3072, hidden_dim: int = 512, num_layers: int = 10, num_classes: int = 10):
        super().__init__()
        assert num_layers >= 8, 'Use at least 8 hidden layers.'
        assert num_layers % 2 == 0, 'For residual model, --layers must be even (each block has 2 dense layers).'
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)

        num_blocks = num_layers // 2
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        nn.init.zeros_(self.classifier.bias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # (N, 3072)
        x = self.input_proj(x)
        x = self.act(x)
        x = self.blocks(x)
        logits = self.classifier(x)
        return logits

# ---------------------- Train / Eval ----------------------

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, running_correct / total


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return running_loss / total, running_correct / total

# ---------------------- Plotting & Logging ----------------------

def save_metrics_csv(stats: List[EpochStats], outdir: str):
    csv_path = os.path.join(outdir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for s in stats:
            writer.writerow([s.epoch, s.train_loss, s.train_acc, s.val_loss, s.val_acc])
    print(f"Saved metrics to {csv_path}")


def plot_curves(stats: List[EpochStats], outdir: str):
    epochs = [s.epoch for s in stats]
    train_loss = [s.train_loss for s in stats]
    val_loss = [s.val_loss for s in stats]
    train_acc = [s.train_acc for s in stats]
    val_acc = [s.val_acc for s in stats]

    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs (Residual MLP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_png = os.path.join(outdir, 'loss_curve.png')
    plt.savefig(loss_png, bbox_inches='tight')
    print(f"Saved {loss_png}")

    plt.figure()
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs (Residual MLP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_png = os.path.join(outdir, 'acc_curve.png')
    plt.savefig(acc_png, bbox_inches='tight')
    print(f"Saved {acc_png}")

# ---------------------- Main ----------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

    # Data: ToTensor() -> [0,1]
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Train/Val split
    val_size = 5000
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.layers % 2 != 0:
        raise ValueError('--layers must be even for the residual model (each block has 2 dense layers).')

    model = DeepResMLP(input_dim=32*32*3, hidden_dim=args.hidden, num_layers=args.layers, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_path = os.path.join(args.outdir, 'best_model.pt')

    history: List[EpochStats] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(EpochStats(epoch, tr_loss, tr_acc, va_loss, va_acc))
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

        # save by minimum validation loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'val_loss': va_loss, 'val_acc': va_acc,
                        'epoch': epoch, 'args': vars(args)}, best_path)
            
    # Final evaluation on test set with best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model from epoch {ckpt['epoch']} "
            f"with val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.4f}")
    else:
        print("Warning: no best checkpoint found; evaluating current model.")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    save_metrics_csv(history, args.outdir)
    plot_curves(history, args.outdir)

if __name__ == '__main__':
    main()
