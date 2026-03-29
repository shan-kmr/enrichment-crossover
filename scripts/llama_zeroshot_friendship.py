#!/usr/bin/env python3
"""Llama 3.1 8B zero-shot eval for Gowalla friendship prediction.

Run on Bouchet HPC with H200 GPU.  Change TIER below per session.
Zero-shot only (no fine-tuning) — FT results already exist.

Usage on HPC:
  TIER=enriched    → reads ~/friendship_data_enriched/,    writes ~/friendship_results/
  TIER=nonenriched → reads ~/friendship_data_nonenriched/, writes ~/friendship_results/
"""

import json
import logging
import os
import torch

logging.disable(logging.WARNING)

from unsloth import FastLanguageModel

# ── CHANGE THIS PER RUN ────────────────────────────────────────────
TIER = os.environ.get("TIER", "enriched")  # "enriched" | "nonenriched"
# ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.expanduser(f"~/friendship_data_{TIER}")
RESULTS_DIR = os.path.expanduser("~/friendship_results")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LEVELS = ["G0", "G1", "G2", "G3", "G4"]

TIER_SUFFIX = "" if TIER == "enriched" else f"_{TIER}"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_accuracy(predictions):
    correct = 0
    total = 0
    for p in predictions:
        pred_text = p["prediction"].strip().lower()
        pred_label = 1 if pred_text.startswith("yes") else 0
        if pred_label == p["label"]:
            correct += 1
        total += 1
    return {"accuracy": round(correct / total, 4) if total else 0, "total": total}


os.makedirs(RESULTS_DIR, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME, max_seq_length=4096,
    load_in_4bit=False, token=HF_TOKEN,
)
FastLanguageModel.for_inference(model)

for level in LEVELS:
    print(f"\n{'=' * 60}")
    print(f"TIER={TIER}  LEVEL={level}")
    print(f"{'=' * 60}")

    test_data = load_jsonl(os.path.join(DATA_DIR, f"test_{level}.jsonl"))
    print(f"Test: {len(test_data)} examples")

    zs_preds = []
    for i, item in enumerate(test_data):
        tok_out = tokenizer.apply_chat_template(
            item["messages"], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        )
        if hasattr(tok_out, "input_ids"):
            input_ids = tok_out.input_ids.to("cuda")
        else:
            input_ids = tok_out.to("cuda")
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=8,
                do_sample=False,
            )
        resp = tokenizer.decode(
            out[0][prompt_len:], skip_special_tokens=True).strip()
        zs_preds.append({
            "label": item["label"],
            "prediction": resp,
        })
        if (i + 1) % 50 == 0:
            print(f"  ZS {i + 1}/{len(test_data)}")

    metrics = compute_accuracy(zs_preds)
    print(f"  ZS Accuracy: {metrics['accuracy']:.4f} ({metrics['total']} samples)")

    out_path = os.path.join(
        RESULTS_DIR, f"llama_zeroshot{TIER_SUFFIX}__{level}.json")
    with open(out_path, "w") as f:
        json.dump(zs_preds, f)
    print(f"  Saved → {out_path}")

print("\nDone!")
