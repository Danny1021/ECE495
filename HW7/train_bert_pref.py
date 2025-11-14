#!/usr/bin/env python3
"""
train_bert_pref.py
Frozen BERT + single linear head trained with a pairwise Bradley–Terry loss.

Now defaults to:
- Clean preset datasets: Anthropic HH (`--preset hh`) or StackExchange Paired (`--preset stack`)
- Pre-tokenization with on-disk caching (no CLI flag needed)
- macOS-friendly DataLoaders (num_workers=0)
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PairItem:
    prompt: str
    text_a: str
    text_b: str
    label: int  # 1 if A preferred, else 0


def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------
# Datasets with fixed schemas
# ---------------------------

class ABPresetDataset(Dataset):
    """
    Clean A/B preference datasets with known schemas.

    Presets:
      - 'hh': Anthropic/hh-rlhf   -> chosen (A), rejected (B)   (splits: train, test)
      - 'stack': lvwerra/stack-exchange-paired -> higher score wins (train only; you can slice)
    Produces PairItem(prompt, text_a, text_b, label) with label=1 if A preferred.
    """
    def __init__(self, preset: str, split: str):
        self.items: List[PairItem] = []

        if preset == "hh":
            # Anthropic/hh-rlhf has 'train' and 'test' only
            split = split.replace("validation", "test", 1)
            ds = load_dataset("Anthropic/hh-rlhf", split=split)
            # columns: chosen, rejected
            for ex in ds:
                a = ex.get("chosen", "")
                b = ex.get("rejected", "")
                if isinstance(a, str) and isinstance(b, str):
                    self.items.append(PairItem(prompt="", text_a=a, text_b=b, label=1))  # A=chosen wins
        elif preset == "stack":
            ds = load_dataset("lvwerra/stack-exchange-paired", split=split)
            # columns: question, response_j, response_k, score_j, score_k
            for ex in ds:
                q = ex.get("question", "") or ""
                aj = ex.get("response_j", "")
                ak = ex.get("response_k", "")
                sj = ex.get("score_j", None)
                sk = ex.get("score_k", None)
                if not (isinstance(aj, str) and isinstance(ak, str)):
                    continue
                try:
                    if sj is None or sk is None or sj == sk:
                        continue
                    if sj > sk:
                        self.items.append(PairItem(prompt=q, text_a=aj, text_b=ak, label=1))
                    else:
                        self.items.append(PairItem(prompt=q, text_a=ak, text_b=aj, label=1))
                except Exception:
                    continue
        else:
            raise ValueError("Unknown preset. Use 'hh' or 'stack'.")

        if len(self.items) == 0:
            raise RuntimeError(f"[ABPresetDataset] No items loaded for preset={preset}, split={split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> PairItem:
        return self.items[idx]


# ---------------------------
# Model (frozen BERT + linear head)
# ---------------------------

class FrozenBertPairwiseHead(nn.Module):
    """
    Freeze the BERT encoder and train a single linear layer on the CLS vector.
    """
    def __init__(self, bert_name="bert-base-uncased"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(bert_name)
        for p in self.enc.parameters():
            p.requires_grad = False
        hidden = self.enc.config.hidden_size
        self.scorer = nn.Linear(hidden, 1)
        nn.init.normal_(self.scorer.weight, std=0.02)
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)

    def forward_score(self, **enc) -> torch.Tensor:
        out = self.enc(**enc, output_hidden_states=False, return_dict=True)
        # Prefer the pooled representation if available (denoised/tanh’d CLS)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            rep = out.pooler_output
        else:
            rep = out.last_hidden_state[:, 0, :]
        return self.scorer(rep).squeeze(-1)


    def forward_pairwise(self, enc_a, enc_b):
        s_a = self.forward_score(**enc_a)
        s_b = self.forward_score(**enc_b)
        return s_a, s_b


def bt_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    # Bradley–Terry / pairwise logistic
    return torch.nn.functional.softplus(-(s_pos - s_neg)).mean()


@torch.no_grad()
def evaluate_pref_accuracy(model: FrozenBertPairwiseHead, dl_val: DataLoader, device="cpu") -> float:
    model.eval()
    correct, total = 0, 0
    for batch in dl_val:
        enc_a = {k: v.to(device) for k, v in batch["enc_a"].items()}
        enc_b = {k: v.to(device) for k, v in batch["enc_b"].items()}
        y = batch["labels"].to(device)
        s_a, s_b = model.forward_pairwise(enc_a, enc_b)
        pred_a = (s_a > s_b).long()  # 1 if A scored higher
        correct += (pred_a == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# ---------------------------
# Pre-tokenization (default)
# ---------------------------

def to_hf(ds: Dataset) -> HFDataset:
    """Convert Dataset[PairItem] to HF Dataset and apply a one-time A/B random swap for balance."""
    rows = []
    for i in range(len(ds)):
        it: PairItem = ds[i]
        # With 50% prob, swap A/B and flip label accordingly (orig label is 1 => A is preferred).
        if random.random() < 0.5:
            rows.append({"prompt": it.prompt, "text_a": it.text_a, "text_b": it.text_b, "label": 1})
        else:
            rows.append({"prompt": it.prompt, "text_a": it.text_b, "text_b": it.text_a, "label": 0})
    return HFDataset.from_list(rows)



def tokenize_pairwise(batch, tokenizer, max_len: int):
    # Concatenate prompt if available (HH uses empty prompts)
    a_texts = [f"[PROMPT]\n{p}\n[RESPONSE]\n{a}" if p else f"[RESPONSE]\n{a}"
               for p, a in zip(batch["prompt"], batch["text_a"])]
    b_texts = [f"[PROMPT]\n{p}\n[RESPONSE]\n{b}" if p else f"[RESPONSE]\n{b}"
               for p, b in zip(batch["prompt"], batch["text_b"])]

    enc_a = tokenizer(a_texts, truncation=True, padding="max_length", max_length=max_len)
    enc_b = tokenizer(b_texts, truncation=True, padding="max_length", max_length=max_len)
    enc_a = {f"a_{k}": v for k, v in enc_a.items()}
    enc_b = {f"b_{k}": v for k, v in enc_b.items()}
    return {**enc_a, **enc_b, "label": batch["label"]}


def build_cached_tokenized_loaders(
    ds_train: Dataset,
    ds_val: Dataset,
    tokenizer,
    max_len: int,
    batch_size: int,
    cache_dir: str = ".cache_tok_pref",
):
    """
    Pre-tokenize both splits once and cache to disk, then create fast DataLoaders.
    """
    os.makedirs(cache_dir, exist_ok=True)

    hf_train = to_hf(ds_train)
    hf_val = to_hf(ds_val)

    tok_fn = lambda batch: tokenize_pairwise(batch, tokenizer, max_len)

    # Use big batch_size for map to speed up; keep num_proc=1 on macOS to avoid fork issues
    hf_train = hf_train.map(tok_fn, batched=True, batch_size=2048,
                            cache_file_name=os.path.join(cache_dir, "train.arrow"))
    hf_val = hf_val.map(tok_fn, batched=True, batch_size=2048,
                        cache_file_name=os.path.join(cache_dir, "val.arrow"))

    keep = ["a_input_ids", "a_attention_mask", "b_input_ids", "b_attention_mask", "label"]
    hf_train = hf_train.remove_columns([c for c in hf_train.column_names if c not in keep])
    hf_val = hf_val.remove_columns([c for c in hf_val.column_names if c not in keep])

    hf_train.set_format(type="torch", columns=keep)
    hf_val.set_format(type="torch", columns=keep)

    def simple_collate(batch):
        return {
            "enc_a": {
                "input_ids": torch.stack([x["a_input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["a_attention_mask"] for x in batch]),
            },
            "enc_b": {
                "input_ids": torch.stack([x["b_input_ids"] for x in batch]),
                "attention_mask": torch.stack([x["b_attention_mask"] for x in batch]),
            },
            "labels": torch.stack([x["label"] for x in batch]).long(),
        }

    # num_workers=0 to avoid fork/tokenizer issues on macOS
    dl_train = DataLoader(hf_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=simple_collate)
    dl_val = DataLoader(hf_val, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=simple_collate)

    # Sanity peek (one batch)
    try:
        first = next(iter(dl_train))
        y0 = first["labels"][:16]
        print(f"[sanity] labels mean={y0.float().mean().item():.3f} sample={y0.tolist()}")
    except Exception as e:
        print("[sanity] skip due to:", e)

    return dl_train, dl_val


# ---------------------------
# Main
# ---------------------------

def _normalize_split_for_preset(preset: str, split: str) -> str:
    if preset == "hh":
        return split.replace("validation", "test", 1)
    return split


def main():
    ap = argparse.ArgumentParser()

    # Preset selection (recommended)
    ap.add_argument("--preset", choices=["hh", "stack"], default=None,
                    help="Use a known clean dataset preset (hh or stack).")

    # Legacy/unused defaults kept for compatibility; ignored when --preset is given
    ap.add_argument("--dataset_mode", choices=["hf", "file"], required=False, default="hf",
                    help="Ignored if --preset is provided.")
    ap.add_argument("--hf_dataset_name", type=str, default="stanfordnlp/SHP")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--data_files", type=str, default=None)
    ap.add_argument("--file_format", type=str, default="json", choices=["json", "csv", "parquet"])
    ap.add_argument("--prompt_col", type=str, default="post")
    ap.add_argument("--a_col", type=str, default="human_ref_A")
    ap.add_argument("--b_col", type=str, default="human_ref_B")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--label_a_wins_value", type=str, default="1")

    # Model / training
    ap.add_argument("--bert_name", type=str, default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-2, help="LR for linear head only.")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)

    # Splits: choose sensible defaults for HH
    ap.add_argument("--train_split", type=str, default="train[:20000]")
    ap.add_argument("--val_split", type=str, default="test[:1000]")

    ap.add_argument("--out_path", type=str, default="bert_pref_head.pt",
                    help="Where to save the trained head+config.")

    args = ap.parse_args()

    if args.preset is None:
        raise ValueError("Please pass --preset hh (Anthropic HH) or --preset stack (StackExchange paired).")

    # Normalize splits for known presets
    train_split = _normalize_split_for_preset(args.preset, args.train_split)
    val_split = _normalize_split_for_preset(args.preset, args.val_split)

    # label_a_wins_value kept for compatibility
    try:
        args.label_a_wins_value = int(args.label_a_wins_value)
    except Exception:
        pass

    set_seed(args.seed)
    device = autodetect_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, use_fast=True)

    # Load preset datasets
    ds_train = ABPresetDataset(preset=args.preset, split=train_split)
    ds_val = ABPresetDataset(preset=args.preset, split=val_split)

    # Pre-tokenize and cache by default (fast epochs)
    dl_train, dl_val = build_cached_tokenized_loaders(
        ds_train=ds_train,
        ds_val=ds_val,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        cache_dir=".cache_tok_pref",  # default cache dir
    )

    model = FrozenBertPairwiseHead(bert_name=args.bert_name).to(device)
    opt = torch.optim.AdamW(model.scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Quick sanity: head trainable, encoder frozen
    # print("enc trainable params:", sum(p.requires_grad for p in model.enc.parameters()))
    # print("head trainable params:", sum(p.requires_grad for p in model.scorer.parameters()))

    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dl_train:
            enc_a = {k: v.to(device) for k, v in batch["enc_a"].items()}
            enc_b = {k: v.to(device) for k, v in batch["enc_b"].items()}
            y = batch["labels"].to(device)

            s_a, s_b = model.forward_pairwise(enc_a, enc_b)
            s_pos = torch.where(y == 1, s_a, s_b)  # A wins when y==1
            s_neg = torch.where(y == 1, s_b, s_a)
            loss = bt_loss(s_pos, s_neg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        val_acc = evaluate_pref_accuracy(model, dl_val, device=device)
        print(f"[Epoch {ep}] train_loss={total_loss/max(1,len(dl_train)):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "bert_name": args.bert_name},
                args.out_path
            )

    print(f"Done. Best val acc: {best_val:.4f}  (saved: {args.out_path})")


if __name__ == "__main__":
    main()
