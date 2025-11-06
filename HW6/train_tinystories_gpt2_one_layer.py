import os, math, argparse, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    pipeline,
)
from torch.optim import AdamW

# ------------------- helpers -------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_one_layer_gpt2(tokenizer, n_head=8, n_embd=512, block_size=256):
    assert n_head >= 8, "Use 8+ attention heads"
    assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=1,
        n_head=n_head,
        n_inner=4*n_embd,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return GPT2LMHeadModel(cfg)

def plot_losses(train_losses, val_losses, out_png):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train CE")
    if val_losses:
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Val CE")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.title("TinyStories • 1-layer GPT-2 • CE vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
    return sum(losses)/len(losses) if losses else float("nan")

def build_hf_tokenized_datasets(tokenizer, block_size=256,
                                max_train_samples=None, max_val_samples=None):
    """
    Loads roneneldan/TinyStories, tokenizes to GPT-2 IDs,
    then groups into contiguous blocks for causal LM training.
    """
    raw = load_dataset("roneneldan/TinyStories")  # splits: train, validation
    val_name = "validation" if "validation" in raw else ("test" if "test" in raw else "train")

    # normalize to a single "text" column
    def to_text(example):
        # Coerce to string; fall back to empty string if missing
        txt = example.get("text") or example.get("story") or ""
        if not isinstance(txt, str):
            txt = str(txt)
        return {"text": txt}

    raw = raw.map(to_text, remove_columns=[c for c in raw["train"].column_names if c != "text"])

    if max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(max_train_samples, len(raw["train"]))))
    if max_val_samples is not None:
        raw[val_name] = raw[val_name].select(range(min(max_val_samples, len(raw[val_name]))))

    def tok(batch):
        # Ensure a clean list[str] with no None values
        texts = batch["text"]
        clean_texts = []
        for t in texts:
            if t is None:
                clean_texts.append("")
            elif isinstance(t, str):
                clean_texts.append(t)
            else:
                clean_texts.append(str(t))
        return tokenizer(clean_texts, add_special_tokens=True)
    
    tokenized = raw.map(tok, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # concatenate then split into blocks
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        result["attention_mask"] = [[1]*block_size for _ in range(len(result["input_ids"]))]
        return result

    lm = tokenized.map(group_texts, batched=True)
    return lm["train"], lm[val_name], raw[val_name]  # tokenized train/val, and raw val for prompts

import torch

@torch.no_grad()
def beam_generate(model, tokenizer, prompts, device,
                  beam_sizes=(1,3,5,10), max_new_tokens=60):
    """
    For each prompt, return top num_beams sequences and a length-normalized
    average log-prob per generated token. Robust to HF versions that don't
    populate `sequences_scores`.
    """
    model.eval()
    rows = []
    for p in prompts:
        row = {"prompt": p, "beams": {}}
        enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)

        for nb in beam_sizes:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=nb,
                num_return_sequences=nb,     # top nb candidates
                do_sample=False,             # pure beam search
                no_repeat_ngram_size=2,
                return_dict_in_generate=True,
                output_scores=True,          # we’ll compute scores below
                pad_token_id=tokenizer.eos_token_id
            )

            seqs = []
            # Preferred path: use sequences_scores if present
            seq_scores = getattr(out, "sequences_scores", None)

            if seq_scores is not None:
                for i in range(out.sequences.size(0)):
                    txt = tokenizer.decode(out.sequences[i], skip_special_tokens=True)
                    seqs.append({"text": txt, "score": float(seq_scores[i].item())})
            else:
                # Fallback: compute avg log-prob/token from transition scores
                # out.scores: list of logits for each generated step
                if out.scores:
                    trans = model.compute_transition_scores(
                        out.sequences, out.scores, normalize_logits=True
                    )  # shape: [num_return_sequences, gen_len]
                    for i in range(out.sequences.size(0)):
                        avg_logprob = float(trans[i].mean().item())
                        txt = tokenizer.decode(out.sequences[i], skip_special_tokens=True)
                        seqs.append({"text": txt, "score": avg_logprob})
                else:
                    # No score info available; return text with None score
                    for i in range(out.sequences.size(0)):
                        txt = tokenizer.decode(out.sequences[i], skip_special_tokens=True)
                        seqs.append({"text": txt, "score": None})

            row["beams"][nb] = seqs
        rows.append(row)
    return rows


# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/one_layer_gpt2")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=2000)
    ap.add_argument("--beam_sizes", type=str, default="1,3,5,10")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.runs_dir, exist_ok=True)

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000  # disable 1024-limit warnings for grouping step
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # datasets (tokenized) + raw val for prompts
    train_ds, val_ds, raw_val = build_hf_tokenized_datasets(
        tokenizer,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    # model
    model = build_one_layer_gpt2(
        tokenizer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size
    ).to(device)

    # optimizer/scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # training
    train_losses, val_losses = [], []
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        tr_ce = running / max(1, len(train_loader))
        va_ce = evaluate(model, val_loader, device)
        train_losses.append(tr_ce); val_losses.append(va_ce)
        print(f"Epoch {ep}: Train CE={tr_ce:.4f} | Val CE={va_ce:.4f} | Val ppl={math.exp(va_ce):.2f}")

        # checkpoints
        ckpt_dir = Path(args.runs_dir) / f"epoch_{ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir.as_posix())
        tokenizer.save_pretrained(ckpt_dir.as_posix())
        if va_ce < best_val:
            best_val = va_ce
            model.save_pretrained((Path(args.runs_dir)/"best").as_posix())
            tokenizer.save_pretrained((Path(args.runs_dir)/"best").as_posix())

    # curves
    plot_png = Path(args.runs_dir) / "loss_curve.png"
    plot_losses(train_losses, val_losses, str(plot_png))
    with open(Path(args.runs_dir)/"losses.csv", "w") as f:
        f.write("epoch,train_ce,val_ce\n")
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{i},{tr:.6f},{va:.6f}\n")

    # prompts from raw validation split
    k = min(5, len(raw_val))
    prompts = [raw_val[i]["text"][:200] for i in random.sample(range(len(raw_val)), k)]

    # beam search with multiple sizes
    beam_sizes = tuple(int(x) for x in args.beam_sizes.split(","))
    gens = beam_generate(model, tokenizer, prompts, device,
                         beam_sizes=beam_sizes, max_new_tokens=args.max_new_tokens)

    with open(Path(args.runs_dir)/"beam_search_generations.txt", "w", encoding="utf-8") as f:
        for ex in gens:
            f.write("=== PROMPT ===\n" + ex["prompt"] + "\n")
            for b, seqs in ex["beams"].items():
                f.write(f"\n--- num_beams={b} (top {b} sequences) ---\n")
                for rank, s in enumerate(seqs, 1):
                    f.write(f"[{rank}] score={s['score']:.4f}\n{s['text']}\n\n")
            f.write("\n" + "="*60 + "\n\n")

    # optional pipeline sanity check
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        device=0 if device.type == "cuda" else -1)
    demo = gen_pipe(prompts[0], max_new_tokens=50, num_beams=3, do_sample=False)
    print("\n[Pipeline check] ", demo[0]["generated_text"])

if __name__ == "__main__":
    main()
