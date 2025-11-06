import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from train_tinystories_gpt2_one_layer import beam_generate  # reuse your fixed version

# --- Paths ---
model_dir = "runs/one_layer_gpt2/best"  # adjust if yours differs

# --- Load model + tokenizer ---
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("Using device:", device)

tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
model.eval()

# --- Define prompts ---
prompts = [
    "Once upon a time there was a little robot who wanted to learn",
    "The sun was shining brightly over the green hills when",
    "In a tiny village by the sea, a young girl discovered"
]

# --- Generate ---
beam_sizes = (1, 3, 5, 10)
gens = beam_generate(model, tokenizer, prompts, device,
                     beam_sizes=beam_sizes, max_new_tokens=60)

# --- Save / print ---
with open("beam_search_generations.txt", "w", encoding="utf-8") as f:
    for ex in gens:
        f.write("=== PROMPT ===\n" + ex["prompt"] + "\n")
        for b, seqs in ex["beams"].items():
            f.write(f"\n--- num_beams={b} (top {b} sequences) ---\n")
            for rank, s in enumerate(seqs, 1):
                f.write(f"[{rank}] score={s['score']}\n{s['text']}\n\n")
        f.write("\n" + "="*60 + "\n\n")

print("Done! See beam_search_generations.txt")
