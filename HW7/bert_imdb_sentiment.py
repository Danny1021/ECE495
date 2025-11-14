#!/usr/bin/env python3
"""
bert_sentiment_analysis.py
Frozen BERT + logistic regression head for IMDB sentiment classification.

Usage (Hugging Face IMDB dataset):
  python bert_sentiment_analysis.py \
    --mode hf \
    --bert_name bert-base-uncased \
    --epochs 2 \
    --batch_size 16 \
    --max_len 256

Usage (GitHub IMDB CSV from Ankit152 repo):
  python bert_sentiment_analysis.py \
    --mode file \
    --data_files /path/to/IMDB_Dataset.csv \
    --file_format csv \
    --text_col review \
    --label_col sentiment \
    --label_pos_value positive \
    --label_neg_value negative \
    --epochs 2 \
    --batch_size 16 \
    --max_len 256
"""

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# -------------------- Utilities --------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ClsItem:
    text: str
    label: int


class IMDBDataset(Dataset):
    """
    Supports:
      - mode='hf': uses Hugging Face 'imdb' dataset
      - mode='file': loads a CSV/JSON dataset (for Ankit152 repo)
    """
    def __init__(
        self,
        mode: str,
        split: str,
        hf_name: str = "imdb",
        data_files: Optional[str] = None,
        file_format: str = "csv",
        text_col: str = "text",
        label_col: str = "label",
        label_pos_value="positive",
        label_neg_value="negative",
        subset_ratio: float = 1.0,
    ):
        if mode == "hf":
            self.ds = load_dataset(hf_name, split=split)
            self.text_col = "text"
            self.label_col = "label"
            self.label_pos_value = 1
            self.label_neg_value = 0
        elif mode == "file":
            if data_files is None:
                raise ValueError("--data_files is required when mode='file'")
            self.ds = load_dataset(file_format, data_files=data_files, split="train")
            self.text_col = text_col
            self.label_col = label_col
            self.label_pos_value = label_pos_value
            self.label_neg_value = label_neg_value
        else:
            raise ValueError("--mode must be 'hf' or 'file'")

        # Optional subsampling
        if subset_ratio < 1.0:
            n = int(len(self.ds) * subset_ratio)
            self.ds = self.ds.select(range(max(1, n)))

        self.index = list(range(len(self.ds)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> ClsItem:
        ex = self.ds[self.index[idx]]
        text = ex[self.text_col]
        y = ex[self.label_col]

        if isinstance(y, str):
            label = 1 if y == self.label_pos_value else 0
        else:
            label = int(y)

        return ClsItem(text=text, label=label)


class SingleTextCollator:
    def __init__(self, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[ClsItem]) -> Dict[str, torch.Tensor]:
        texts = [bi.text for bi in batch]
        labels = torch.tensor([bi.label for bi in batch], dtype=torch.float32)
        enc = self.tok(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {"enc": enc, "labels": labels}


# -------------------- Model --------------------

class FrozenBertSentimentHead(nn.Module):
    """
    Frozen BERT encoder + single-logit linear head on CLS token.
    """
    def __init__(self, bert_name="bert-base-uncased"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(bert_name)
        for p in self.enc.parameters():
            p.requires_grad = False
        hidden = self.enc.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)
        nn.init.normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward_logits(self, **enc) -> torch.Tensor:
        device_type = enc["input_ids"].device.type
        use_autocast = device_type in ("mps", "cuda")
        autocast_dtype = torch.float16 if device_type == "mps" else torch.bfloat16
        ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if use_autocast else contextlib.nullcontext()
        with ctx:
            out = self.enc(**enc, return_dict=True)
            cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls.float()).squeeze(-1)
        return logits


# -------------------- Train / Eval --------------------

def train_bert_head(
    train_dl: DataLoader,
    val_dl: DataLoader,
    bert_name="bert-base-uncased",
    epochs=2,
    lr=1e-3,
    weight_decay=0.0,
    device="cpu",
):
    model = FrozenBertSentimentHead(bert_name=bert_name).to(device)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_dl))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(10, total_steps // 20), num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()

    best_val = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            enc = {k: v.to(device) for k, v in batch["enc"].items()}
            y = batch["labels"].to(device)

            logits = model.forward_logits(**enc)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()

        val_acc = evaluate_accuracy(model, val_dl, device)
        print(f"[Epoch {ep}] train_loss={total_loss/max(1,len(train_dl)):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def evaluate_accuracy(model: FrozenBertSentimentHead, dl: DataLoader, device="cpu") -> float:
    model.eval()
    preds, gts = [], []
    for batch in dl:
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        y = batch["labels"].to(device)
        logits = model.forward_logits(**enc)
        pred = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
        preds.extend(pred)
        gts.extend(y.long().cpu().tolist())
    correct = sum(int(p == t) for p, t in zip(preds, gts))
    return correct / max(1, len(gts))


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf", "file"], default="hf",
                        help="Use HuggingFace IMDB (hf) or a file (file).")
    parser.add_argument("--hf_name", type=str, default="imdb")
    parser.add_argument("--data_files", type=str, default=None)
    parser.add_argument("--file_format", type=str, default="csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--label_pos_value", type=str, default="positive")
    parser.add_argument("--label_neg_value", type=str, default="negative")
    parser.add_argument("--subset_ratio", type=float, default=1.0)

    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    device = autodetect_device()
    print("Using device:", device)

    # Load datasets
    if args.mode == "hf":
        ds_train_full = IMDBDataset("hf", "train", hf_name=args.hf_name, subset_ratio=args.subset_ratio)
        ds_test = IMDBDataset("hf", "test", hf_name=args.hf_name, subset_ratio=args.subset_ratio)
    else:
        ds_all = IMDBDataset(
            "file", "train",
            data_files=args.data_files,
            file_format=args.file_format,
            text_col=args.text_col,
            label_col=args.label_col,
            label_pos_value=args.label_pos_value,
            label_neg_value=args.label_neg_value,
            subset_ratio=args.subset_ratio,
        )
        n_test = max(1000, len(ds_all) // 10)
        n_train_full = max(1, len(ds_all) - n_test)
        ds_train_full, ds_test = random_split(ds_all, [n_train_full, n_test],
                                              generator=torch.Generator().manual_seed(args.seed))

    # Train/val split
    n_val = max(1000, int(len(ds_train_full) * 0.05)) if len(ds_train_full) > 2000 else max(100, int(len(ds_train_full)*0.05))
    n_val = min(n_val, max(1, len(ds_train_full)//5))
    n_train = max(1, len(ds_train_full) - n_val)
    ds_train, ds_val = random_split(ds_train_full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    tok = AutoTokenizer.from_pretrained(args.bert_name, use_fast=True)
    collate = SingleTextCollator(tok, max_len=args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    model, best_val = train_bert_head(
        dl_train, dl_val,
        bert_name=args.bert_name,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    test_acc = evaluate_accuracy(model, dl_test, device=device)
    print(f"\n[BERT+LogReg CLS] Best Val Acc: {best_val:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
