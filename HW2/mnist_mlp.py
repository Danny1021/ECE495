# mnist_mlp.py
import argparse, random, csv, itertools, math
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms

# --- Repro ---
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model: MLP with ReLU hidden layers and LogSoftmax output ---
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int = 10, p_drop: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if p_drop > 0:
                layers += [nn.Dropout(p_drop)]
            prev = h
        layers += [nn.Linear(prev, num_classes), nn.LogSoftmax(dim=1)]  # output is log-probabilities
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 784)
        return self.net(x)         # (B, 10) log-probs

# --- Datasets / Transforms ---
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

def load_mnist_raw(train: bool, transform) -> Dataset:
    return datasets.MNIST(root="./data", train=train, download=True, transform=transform)

def compute_mean_image(dataset: Dataset, batch_size: int = 1024) -> torch.Tensor:
    """
    Compute μ = (1/N) Σ x_i as a mean image (shape (1,28,28)) over the given dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    n = 0
    mean_sum = torch.zeros(1, 28, 28)
    with torch.no_grad():
        for x, _ in loader:
            # x in [0,1], shape (B,1,28,28)
            mean_sum += x.sum(dim=0)
            n += x.size(0)
    return mean_sum / max(n, 1)

class GlobalMeanShifted(Dataset):
    """Wraps a dataset so each sample becomes x - μ_image."""
    def __init__(self, base: Dataset, mean_image: torch.Tensor):
        self.base = base
        self.mean_image = mean_image

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x - self.mean_image, y

def build_train_val_splits(seed: int, val_fraction: float, base_transform) -> Tuple[Subset, Subset]:
    full = load_mnist_raw(train=True, transform=base_transform)
    val_size = int(len(full) * val_fraction)
    train_size = len(full) - val_size
    g = torch.Generator().manual_seed(seed)
    return random_split(full, [train_size, val_size], generator=g)

def make_datasets(
    transform_mode: str,
    val_fraction: float,
    seed: int,
    mean_scope: str = "train"  # "train" or "all" (assignment wording says whole dataset; beware leakage)
):
    """
    transform_mode:
      - "none"        : ToTensor only
      - "default"     : ToTensor + Normalize(MNIST_MEAN, MNIST_STD)
      - "global-mean" : subtract μ image (computed from TRAIN or ALL)
    mean_scope: controls where μ is computed from if transform_mode == "global-mean".
    """
    # Start from raw tensors (no normalization) to define a deterministic split
    raw_transform = transforms.ToTensor()
    train_raw, val_raw = build_train_val_splits(seed, val_fraction, raw_transform)
    test_raw = load_mnist_raw(train=False, transform=raw_transform)

    if transform_mode == "none":
        return train_raw, val_raw, test_raw, None

    if transform_mode == "default":
        norm = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        base_train = load_mnist_raw(train=True, transform=transforms.Compose([transforms.ToTensor(), norm]))
        # Re-map the same indices onto a normalized underlying dataset
        train_ds = Subset(base_train, train_raw.indices)
        val_ds   = Subset(base_train, val_raw.indices)
        test_ds  = load_mnist_raw(train=False, transform=transforms.Compose([transforms.ToTensor(), norm]))
        return train_ds, val_ds, test_ds, None

    # if transform_mode == "global-mean":
    #     # Compute μ image from requested scope
    #     if mean_scope == "all":
    #         # WARNING: uses test images for μ (data leakage), but aligns with assignment wording "total set"
    #         total_for_mean = torch.utils.data.ConcatDataset([load_mnist_raw(True, raw_transform),
    #                                                          load_mnist_raw(False, raw_transform)])
    #     else:
    #         total_for_mean = load_mnist_raw(True, raw_transform)
    #     mu = compute_mean_image(total_for_mean)  # (1,28,28)

    #     return (GlobalMeanShifted(train_raw, mu),
    #             GlobalMeanShifted(val_raw,   mu),
    #             GlobalMeanShifted(test_raw,  mu),
    #             mu)
    
    if transform_mode == "global-mean":
        # Leakage-free: compute μ from the TRAIN SUBSET ONLY (train_raw)
        # train_raw is a Subset of the official training set with ToTensor() only
        mu = compute_mean_image(train_raw)  # (1, 28, 28)

        return (
            GlobalMeanShifted(train_raw, mu),
            GlobalMeanShifted(val_raw,   mu),
            GlobalMeanShifted(test_raw,  mu),
            mu
        )


    raise ValueError(f"Unknown transform_mode: {transform_mode}")

# --- Train / Eval ---
def make_loader(ds, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def make_optimizer(model, opt_name: str, lr: float, weight_decay: float, momentum: float):
    # "Gradient descent" in practice = mini-batch SGD variants; we expose SGD and Adam.
    if opt_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("--opt must be 'sgd' or 'adam'")

def make_scheduler(optimizer, name: str, step_size: int, gamma: float):
    name = name.lower()
    if name == "none":
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=2)
    raise ValueError("--scheduler must be none|steplr|plateau")

def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        log_probs = model(x)
        loss = F.nll_loss(log_probs, y)   # matches LogSoftmax output

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (log_probs.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    return run_epoch(model, loader, device, optimizer=None)

def train_once(
    hidden_sizes: List[int],
    transform_mode: str,
    val_fraction: float,
    seed: int,
    global_mean_scope: str,
    batch_size: int,
    epochs: int,
    lr: float,
    wd: float,
    momentum: float,
    opt: str,
    scheduler_name: str,
    step_size: int,
    gamma: float,
    dropout: float,
    device
) -> Dict[str, Any]:
    train_ds, val_ds, test_ds, mu = make_datasets(transform_mode, val_fraction, seed, global_mean_scope)
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False)

    model = MLP(28*28, hidden_sizes, 10, p_drop=dropout).to(device)
    optimizer = make_optimizer(model, opt, lr, wd, momentum)
    scheduler = make_scheduler(optimizer, scheduler_name, step_size, gamma)

    best_val, best_epoch = -1.0, -1
    hist = []

    for ep in range(1, epochs + 1):

        curr_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device)

        if scheduler_name == "plateau":
            scheduler.step(va_loss)
        elif scheduler_name == "steplr":
            scheduler.step()

        if va_acc > best_val:
            best_val, best_epoch = va_acc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        hist.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc})
        print(f"[{transform_mode:<11}] hs={hidden_sizes} lr {curr_lr:.4f} ep {ep:02d}/{epochs} "
              f"train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}")

    # Load best and test
    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"Best val acc {best_val:.4f} @ epoch {best_epoch}; Test acc {te_acc:.4f}")

    return {
        "hidden_sizes": hidden_sizes,
        "transform": transform_mode,
        "val_acc_best": best_val,
        "best_epoch": best_epoch,
        "test_acc": te_acc,
        "history": hist,
    }

# --- CLI / Sweeps ---
def main():
    ap = argparse.ArgumentParser("MNIST MLP (L=1..3) with togglable transforms and normalization (global-mean).")
    ap.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 128],
                    help="Explicit hidden sizes for a single run, e.g. --hidden-sizes 128 64")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--opt", choices=["sgd", "adam"], default="adam",
                    help="Mini-batch gradient descent variant")
    ap.add_argument("--scheduler", choices=["none", "steplr", "plateau"], default="none")
    ap.add_argument("--step-size", type=int, default=0.1, help="StepLR step_size")
    ap.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--val-fraction", type=float, default=0.1)
    # Transforms: none|default|global-mean
    ap.add_argument("--transform", choices=["none", "default", "global-mean"], default="none")
    ap.add_argument("--global-mean-scope", choices=["train", "all"], default="train",
                    help="Where to compute μ from if using global-mean; 'all' follows assignment wording but leaks info")

    # Convenience: compare normalization (same architecture)
    ap.add_argument("--compare-normalization", action="store_true",
                    help="Run two models: transform=none vs transform=global-mean (μ from --global-mean-scope)")

    # Grid sweep across depths (L) and a set of widths (same width repeated L times)
    ap.add_argument("--sweep", action="store_true", help="Run a grid over depths and widths")
    ap.add_argument("--layers", type=int, nargs="*", default=[1, 2, 3], help="Depths to try (hidden layer counts)")
    ap.add_argument("--widths", type=int, nargs="*", default=[64, 128, 256], help="Widths to try per layer")
    ap.add_argument("--results-csv", type=str, default="", help="Optional path to save a CSV of results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results: List[Dict[str, Any]] = []

    if args.compare_normalization:
        # As per assignment: compare accuracy with and without μ subtraction for one architecture
        for mode in ["none", "global-mean"]:
            res = train_once(
                hidden_sizes=args.hidden_sizes,
                transform_mode=mode,
                val_fraction=args.val_fraction,
                seed=args.seed,
                global_mean_scope=args.global_mean_scope,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                wd=args.weight_decay,
                momentum=args.momentum,
                opt=args.opt,
                scheduler_name=args.scheduler,
                step_size=args.step_size,
                gamma=args.gamma,
                dropout=args.dropout,
                device=device,
            )
            results.append(res)

    elif args.sweep:
        # L in {1,2,3}, width in set; hidden sizes = [width]*L
        for L, W in itertools.product(args.layers, args.widths):
            hs = [W] * L
            res = train_once(
                hidden_sizes=hs,
                transform_mode=args.transform,
                val_fraction=args.val_fraction,
                seed=args.seed,
                global_mean_scope=args.global_mean_scope,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                wd=args.weight_decay,
                momentum=args.momentum,
                opt=args.opt,
                scheduler_name=args.scheduler,
                step_size=args.step_size,
                gamma=args.gamma,
                dropout=args.dropout,
                device=device,
            )
            results.append(res)

    else:
        # Single run
        res = train_once(
            hidden_sizes=args.hidden_sizes,
            transform_mode=args.transform,
            val_fraction=args.val_fraction,
            seed=args.seed,
            global_mean_scope=args.global_mean_scope,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.weight_decay,
            momentum=args.momentum,
            opt=args.opt,
            scheduler_name=args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
            dropout=args.dropout,
            device=device,
        )
        results.append(res)

    # Print compact summary
    print("\n=== Summary ===")
    for r in results:
        print(f"hs={r['hidden_sizes']}, transform={r['transform']:>11}, "
              f"best_val={r['val_acc_best']:.4f}, test={r['test_acc']:.4f}, best_ep={r['best_epoch']}")

    # Optional CSV
    if args.results_csv:
        with open(args.results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["hidden_sizes", "transform", "val_acc_best", "test_acc", "best_epoch"])
            w.writeheader()
            for r in results:
                w.writerow({
                    "hidden_sizes": "-".join(map(str, r["hidden_sizes"])),
                    "transform": r["transform"],
                    "val_acc_best": f"{r['val_acc_best']:.4f}",
                    "test_acc": f"{r['test_acc']:.4f}",
                    "best_epoch": r["best_epoch"],
                })

if __name__ == "__main__":
    main()
