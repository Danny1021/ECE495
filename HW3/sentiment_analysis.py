#!/usr/bin/env python3
"""
Average-Embedding Sentiment Classifier (IMDB, Binary)

Single-run mode (no internal sweeping):
- Fixed dataset path/columns on your machine:
    * /Users/dany/Desktop/ECE495/HW3/IMDB-Dataset.csv
    * columns: review, sentiment
- Learns token embeddings end-to-end and averages them per review.
- Choose exactly ONE configuration per run:
    * --model logreg OR --model mlp
    * --emb_dim <int>
    * If MLP: --mlp_arch <comma-separated hidden sizes> (e.g., 128,64)
- Prints per-epoch validation metrics and final test metrics.

Example usage
-------------
# Logistic regression with 200-d embeddings
python avg_embed_sentiment.py --model logreg --emb_dim 200

# MLP with two hidden layers 128→64 and 300-d embeddings
python avg_embed_sentiment.py --model mlp --emb_dim 300 --mlp_arch 128,64
"""
from __future__ import annotations
import argparse
import html
import os
import random
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------
# Fixed data config (per your request)
# ------------------------
DATA_CSV = "/Users/dany/Desktop/ECE495/HW3/IMDB-Dataset.csv"
TEXT_COL = "review"
LABEL_COL = "sentiment"

# ------------------------
# CLI
# ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Avg-Embedding IMDB Sentiment (single-run)")
    # run choice
    p.add_argument('--model', choices=['logreg','mlp'], required=True)
    p.add_argument('--emb_dim', type=int, required=True, help='Embedding dimension for token vectors')
    p.add_argument('--mlp_arch', type=str, default=None, help='Comma-separated hidden sizes for MLP, e.g., 128,64')

    # training knobs
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=2)

    # preprocessing
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--min_freq', type=int, default=2)

    # device
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # optional save
    p.add_argument('--save_results', action='store_true', help='If set, save a one-row CSV of test metrics')
    p.add_argument('--results_dir', type=str, default='runs', help='Directory for results if saving')
    return p.parse_args()

# ------------------------
# Data utilities
# ------------------------
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

def basic_tokenize(text: str) -> List[str]:
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    return TOKEN_PATTERN.findall(text)

class Vocab:
    def __init__(self, stoi: Dict[str,int], itos: List[str], pad_idx: int, unk_idx: int):
        self.stoi, self.itos = stoi, itos
        self.pad_idx, self.unk_idx = pad_idx, unk_idx
    @classmethod
    def build(cls, corpus_tokens: List[List[str]], min_freq: int=2) -> 'Vocab':
        from collections import Counter
        freq = Counter()
        for toks in corpus_tokens:
            freq.update(toks)
        itos = ['<pad>', '<unk>']
        for tok, c in freq.most_common():
            if c >= min_freq and tok not in itos:
                itos.append(tok)
        stoi = {t:i for i,t in enumerate(itos)}
        return cls(stoi, itos, pad_idx=0, unk_idx=1)
    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        return [ self.stoi.get(t, self.unk_idx) for t in tokens[:max_len] ]

class ReviewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocab, max_len: int):
        self.vocab, self.max_len = vocab, max_len
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.ids = [vocab.encode(basic_tokenize(t), max_len) for t in texts]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.ids[idx], self.labels[idx]

def collate_fn(batch, pad_idx: int):
    ids_list, labels = zip(*batch)
    max_len = max(len(x) for x in ids_list) if ids_list else 1
    padded = torch.full((len(ids_list), max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((len(ids_list), max_len), dtype=torch.bool)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        if L:
            padded[i, :L] = torch.tensor(ids, dtype=torch.long)
            mask[i, :L] = True
    return padded, mask, torch.stack(labels)

# ------------------------
# Models
# ------------------------
class AvgEmbedBase(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    def forward_avg(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        E = self.emb(ids)
        mask_f = mask.float().unsqueeze(-1)
        summed = (E * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / counts

class AvgEmbedLogReg(AvgEmbedBase):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, 1)
    def forward(self, ids, mask):
        return self.fc(self.forward_avg(ids, mask)).squeeze(-1)

class AvgEmbedDeepMLP(AvgEmbedBase):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, layers: int, dropout: float=0.2):
        super().__init__(vocab_size, emb_dim)
        blocks = [nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(layers-1):
            blocks += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        blocks += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*blocks)
    def forward(self, ids, mask):
        return self.net(self.forward_avg(ids, mask)).squeeze(-1)

class AvgEmbedArchiMLP(AvgEmbedBase):
    """MLP head with an explicit per-layer architecture, e.g., [128, 64]."""
    def __init__(self, vocab_size: int, emb_dim: int, hidden_layers: list[int], dropout: float=0.2):
        super().__init__(vocab_size, emb_dim)
        dims = [emb_dim] + list(hidden_layers)
        blocks = []
        for i in range(len(dims)-1):
            blocks += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        blocks += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*blocks)
    def forward(self, ids, mask):
        return self.net(self.forward_avg(ids, mask)).squeeze(-1)

# ------------------------
# Train / Eval
# ------------------------
@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float,float]:
    model.eval()
    all_logits, all_labels = [], []
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits) if all_logits else torch.zeros(0)
    labels = torch.cat(all_labels) if all_labels else torch.zeros(0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = accuracy_score(labels.numpy(), preds.numpy())
    f1 = f1_score(labels.numpy(), preds.numpy())
    return acc, f1


def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str,
              epochs: int, lr: float, patience: int) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val, best_state, no_improve = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            loss = loss_fn(model(ids, mask), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        val_acc, val_f1 = eval_model(model, val_loader, device)
        print(f"Epoch {ep:02d} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
        if val_acc > best_val:
            best_val, best_state, no_improve = val_acc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    if best_state:
        model.load_state_dict(best_state)

# ------------------------
# Main
# ------------------------

def load_imdb_fixed() -> Tuple[List[str], List[int]]:
    df = pd.read_csv(DATA_CSV)
    assert TEXT_COL in df.columns and LABEL_COL in df.columns, f"CSV must have columns '{TEXT_COL}' and '{LABEL_COL}'. Got: {list(df.columns)}"
    texts = df[TEXT_COL].astype(str).tolist()
    raw = df[LABEL_COL].astype(str).str.strip().str.lower().tolist()
    labels = [1 if (y.startswith('pos') or y=='1' or y=='true') else 0 for y in raw]
    # quick sanity
    pos = sum(labels); neg = len(labels)-pos
    print(f"Loaded {len(labels)} rows from {DATA_CSV} | pos={pos} neg={neg}")
    return texts, labels


def _parse_arch_string(arch_string: str) -> list[int]:
    parts = [p.strip() for p in arch_string.split(',') if p.strip()]
    if not parts:
        raise ValueError("--mlp_arch was provided but empty. Example: --mlp_arch 128,64")
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid --mlp_arch '{arch_string}'. Use comma-separated ints, e.g., 128,64")
    


def main():
    args = parse_args()
    if args.model == 'mlp' and not args.mlp_arch:
        raise SystemExit("For --model mlp you must provide --mlp_arch, e.g., --mlp_arch 128,64")

    texts, labels = load_imdb_fixed()
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    vocab = Vocab.build([basic_tokenize(t) for t in X_train], min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab.itos)} | min_freq={args.min_freq}")

    collate = lambda b: collate_fn(b, vocab.pad_idx)
    dl_train = DataLoader(ReviewsDataset(X_train, y_train, vocab, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dl_val   = DataLoader(ReviewsDataset(X_val, y_val, vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dl_test  = DataLoader(ReviewsDataset(X_test, y_test, vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Build model
    if args.model == 'logreg':
        print(f"=== CONFIG | model=logreg | emb_dim={args.emb_dim} ===")
        model = AvgEmbedLogReg(vocab_size=len(vocab.itos), emb_dim=args.emb_dim)
    else:
        arch = _parse_arch_string(args.mlp_arch)
        print(f"=== CONFIG | model=mlp | emb_dim={args.emb_dim} | arch={arch} ===")
        model = AvgEmbedArchiMLP(vocab_size=len(vocab.itos), emb_dim=args.emb_dim, hidden_layers=arch, dropout=args.dropout)

    # Train (prints per-epoch val metrics)
    train_one(model, dl_train, dl_val, args.device, args.epochs, args.lr, args.patience)

    # Final test metrics
    test_acc, test_f1 = eval_model(model, dl_test, args.device)
    print(f"=== Test === \n acc={test_acc:.4f} | f1={test_f1:.4f}")

    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, 'single_run_results.csv')
        import pandas as pd
        row = {
            'model': args.model,
            'emb_dim': args.emb_dim,
            'arch': args.mlp_arch if args.mlp_arch else '-',
            'test_acc': test_acc,
            'test_f1': test_f1,
        }
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print('Saved:', out_path)

if __name__ == '__main__':
    main()