#!/usr/bin/env python3
# eval_tinystories_with_bert_pref.py
# Compare beam vs. nucleus generations from your local TinyStories GPT using a saved BERT preference head.

import os, csv, argparse
import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import numpy as np

# --------------------- utils ---------------------

def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_matmul_precision_for_mps():
    # helps on Apple Silicon
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# --------------------- BERT pref head (matches your training code) ---------------------

class FrozenBertPairwiseHead(nn.Module):
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
        # prefer pooler_output if available
        rep = out.pooler_output if getattr(out, "pooler_output", None) is not None \
              else out.last_hidden_state[:, 0, :]
        return self.scorer(rep).squeeze(-1)

    def forward_pairwise(self, enc_a, enc_b):
        s_a = self.forward_score(**enc_a)
        s_b = self.forward_score(**enc_b)
        return s_a, s_b

def load_bert_pref(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")    
    bert_name = ckpt.get("bert_name", "bert-base-uncased")
    tok = AutoTokenizer.from_pretrained(bert_name, use_fast=True)
    model = FrozenBertPairwiseHead(bert_name)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    return model, tok

@torch.no_grad()
def bert_pref_rate_and_margin(
    model: FrozenBertPairwiseHead,
    tok,
    prompts: List[str],
    beams: List[str],
    nucs: List[str],
    max_len: int,
    batch_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    assert len(prompts) == len(beams) == len(nucs)
    prefs, margins = [], []

    def chunk(n):
        for i in range(0, len(prompts), n):
            yield i, i+n

    for i0, i1 in chunk(batch_size):
        P = prompts[i0:i1]
        A = beams[i0:i1]
        B = nucs[i0:i1]
        txt_a = [f"[PROMPT]\n{p}\n[RESPONSE]\n{a}" if p else f"[RESPONSE]\n{a}" for p,a in zip(P,A)]
        txt_b = [f"[PROMPT]\n{p}\n[RESPONSE]\n{b}" if p else f"[RESPONSE]\n{b}" for p,b in zip(P,B)]
        enc_a = tok(txt_a, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc_b = tok(txt_b, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc_a = {k: v.to(device) for k,v in enc_a.items()}
        enc_b = {k: v.to(device) for k,v in enc_b.items()}
        s_a, s_b = model.forward_pairwise(enc_a, enc_b)
        prefs.append((s_a > s_b).cpu())
        margins.append((s_a - s_b).cpu())

    pref_mask = torch.cat(prefs)
    margin = torch.cat(margins)
    rate_beam = pref_mask.float().mean().item() if pref_mask.numel() else 0.0
    return pref_mask, margin, rate_beam

# --------------------- GPT generation ---------------------

def load_local_gpt(model_dir: str, device: str):
    gpt_tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # GPT-2 often lacks a pad token; set pad -> eos to avoid warnings
    if gpt_tok.pad_token_id is None and gpt_tok.eos_token_id is not None:
        gpt_tok.pad_token = gpt_tok.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(model_dir)
    gpt.eval().to(device)
    return gpt, gpt_tok

@torch.no_grad()
def generate_beam(gpt, tok, prompts: List[str], max_new_tokens=80, num_beams=4, device="cpu"):
    inputs = tok(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(device)
    gen = gpt.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    # decode only the generated continuation (simple: decode full and strip the prompt)
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    # strip the prompt prefix to get the continuation text
    conts = []
    for prompt, full in zip(prompts, outs):
        conts.append(full[len(prompt):].strip() if full.startswith(prompt) else full.strip())
    return conts

@torch.no_grad()
def generate_nucleus(gpt, tok, prompts: List[str], max_new_tokens=80, top_p=0.9, temperature=0.8, device="cpu"):
    inputs = tok(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(device)
    gen = gpt.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    conts = []
    for prompt, full in zip(prompts, outs):
        conts.append(full[len(prompt):].strip() if full.startswith(prompt) else full.strip())
    return conts

# --------------------- main ---------------------

def load_prompts(path: str | None) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln]
    # TinyStories-style default prompts if no file provided
    return [
        "Once upon a time, there was a tiny robot who wanted to learn.",
        "In a small village, a young fox found a shiny key near the river.",
        "A little cloud wanted to see what lay beyond the mountains.",
        "Timmy built a paper rocket and dreamed of the stars.",
        "A ladybug discovered a secret garden behind a blue door.",
        "A turtle wondered how music sounds under the sea.",
        "A candle and a moth became friends on a windy night.",
        "Two droplets raced down a window during a storm.",
        "A tiny witch tried her very first spell.",
        "A toy train woke up in a quiet attic."
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt_model_dir", type=str, required=True,
                    help="Path to your local TinyStories GPT model directory (contains config.json, model.safetensors, tokenizer files).")
    ap.add_argument("--bert_ckpt", type=str, default="bert_pref_head_hh_20k.pt",
                    help="Path to your saved BERT preference head .pt file.")
    ap.add_argument("--prompts_file", type=str, default=None,
                    help="Optional path to a newline-delimited prompts file.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional path to write per-example results CSV.")
    # generation params
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    # BERT scoring params
    ap.add_argument("--bert_max_len", type=int, default=160)
    ap.add_argument("--bert_batch_size", type=int, default=64)
    args = ap.parse_args()

    device = autodetect_device()
    set_matmul_precision_for_mps()
    print(f"Using device: {device}")

    # Load models
    gpt, gpt_tok = load_local_gpt(args.gpt_model_dir, device)
    bert_model, bert_tok = load_bert_pref(args.bert_ckpt, device)

    # Prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts.")

    # Generate
    beam_texts = generate_beam(
        gpt, gpt_tok, prompts,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    nuc_texts = generate_nucleus(
        gpt, gpt_tok, prompts,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        device=device,
    )

    # Score with BERT pref head
    mask, margin, rate_beam = bert_pref_rate_and_margin(
        bert_model, bert_tok,
        prompts, beam_texts, nuc_texts,
        max_len=args.bert_max_len,
        batch_size=args.bert_batch_size,
        device=device
    )

    n = len(prompts)
    print(f"\nBERT prefers BEAM in {rate_beam*100:.1f}% of cases ({int(mask.sum().item())}/{n}).")
    print("Median margin (s_beam - s_nucleus): {:.3f}".format(torch.median(margin).item()))

    # Optional CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prompt", "beam", "nucleus", "bert_pref", "margin"])
            for p, b, u, m, d in zip(prompts, beam_texts, nuc_texts, mask.tolist(), margin.tolist()):
                w.writerow([p, b, u, "beam" if m else "nucleus", f"{d:.4f}"])
        print(f"Wrote results to {args.out_csv}")

    # --- Optional Visualization ---
    plt.style.use("ggplot")

    plt.figure(figsize=(7,5))
    plt.hist(margin.numpy(), bins=40, color='gray', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Distribution of BERT Preference Margins")
    plt.xlabel("s_beam - s_nucleus (positive → prefers beam)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("bert_pref_margin_hist.png")
    print("Saved plot: bert_pref_margin_hist.png")


if __name__ == "__main__":
    main()
