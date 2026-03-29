#!/usr/bin/env python3
"""Llama 3.1 8B fine-tuning + zero-shot eval for NYC taxi tasks.

Run on Bouchet HPC with H200 GPU.  Change TIER and TASK below per session.
Each level: zero-shot eval → fine-tune on train → fine-tuned eval.

Usage on HPC:
  TIER=latlng   TASK=ranking  → reads ~/nyc_taxi_ranking_latlng/
  TIER=zone_id  TASK=duration → reads ~/nyc_taxi_duration_zone_id/
  etc.
"""

import json
import logging
import os
import random
import re
import torch
from collections import defaultdict

logging.disable(logging.WARNING)

from unsloth import FastLanguageModel

# ── CHANGE THESE PER RUN ──────────────────────────────────────────
TIER = "latlng"     # "latlng" | "zone_id" | "enriched"
TASK = "ranking"    # "ranking" | "duration"
# ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.expanduser(f"~/nyc_taxi_{TASK}_{TIER}")
RESULTS_DIR = os.path.expanduser(f"~/nyc_taxi_results_{TASK}_{TIER}")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LEVELS = ["G0", "G1", "G2", "G3", "G4"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_ranking_metrics(predictions):
    origin_preds = defaultdict(list)
    for p in predictions:
        s = 1.0 if p["prediction"].strip().lower().startswith("yes") else 0.0
        origin_preds[p["origin_zone"]].append((s, p["label"]))

    acc1, acc5, mrr = [], [], []
    for oid, preds in origin_preds.items():
        rng = random.Random(oid)
        ranked = sorted(preds, key=lambda x: (-x[0], rng.random()))
        for rank, (_, label) in enumerate(ranked, 1):
            if label == 1:
                acc1.append(float(rank == 1))
                acc5.append(float(rank <= 5))
                mrr.append(1.0 / rank)
                break
    n = len(acc1)
    return {
        "acc_at_1": round(sum(acc1) / n, 4) if n else 0,
        "acc_at_5": round(sum(acc5) / n, 4) if n else 0,
        "mrr": round(sum(mrr) / n, 4) if n else 0,
    }


def compute_duration_metrics(predictions):
    errors = []
    for p in predictions:
        try:
            pred_val = float(re.findall(r"[\d.]+", p["prediction"])[0])
        except (IndexError, ValueError):
            pred_val = 15.0  # fallback
        errors.append(abs(pred_val - p["duration_min"]))

    import numpy as np
    errors = np.array(errors)
    return {
        "mae": round(float(errors.mean()), 4),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 4),
    }


def run_inference(model, tokenizer, test_data, task, tag="ZS"):
    preds = []
    max_new = 8 if task == "ranking" else 16
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
                input_ids=input_ids, max_new_tokens=max_new, do_sample=False)
        resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        pred_item = {"prediction": resp}
        if task == "ranking":
            pred_item.update({
                "origin_zone": item["origin_zone"],
                "candidate_zone": item["candidate_zone"],
                "label": item["label"],
            })
        else:
            pred_item.update({
                "duration_min": item["duration_min"],
                "PULocationID": item["PULocationID"],
                "DOLocationID": item["DOLocationID"],
            })
        preds.append(pred_item)

        if (i + 1) % 200 == 0:
            print(f"  {tag} {i + 1}/{len(test_data)}")

    return preds


os.makedirs(RESULTS_DIR, exist_ok=True)

for level in LEVELS:
    print(f"\n{'=' * 60}")
    print(f"TIER={TIER}  TASK={TASK}  LEVEL={level}")
    print(f"{'=' * 60}")

    train_data = load_jsonl(os.path.join(DATA_DIR, f"train_{level}.jsonl"))
    test_data = load_jsonl(os.path.join(DATA_DIR, f"test_{level}.jsonl"))
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # ── load base model ────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=8192, load_in_4bit=False, token=HF_TOKEN)

    # ── zero-shot ──────────────────────────────────────────────────
    print("Zero-shot inference ...")
    FastLanguageModel.for_inference(model)
    zs_preds = run_inference(model, tokenizer, test_data, TASK, "ZS")

    if TASK == "ranking":
        zs_m = compute_ranking_metrics(zs_preds)
        print(f"  ZS: Acc@1={zs_m['acc_at_1']:.4f}  "
              f"Acc@5={zs_m['acc_at_5']:.4f}  MRR={zs_m['mrr']:.4f}")
    else:
        zs_m = compute_duration_metrics(zs_preds)
        print(f"  ZS: MAE={zs_m['mae']:.4f}  RMSE={zs_m['rmse']:.4f}")

    with open(os.path.join(
            RESULTS_DIR, f"llama_zeroshot_{TIER}__{TASK}_{level}.json"), "w") as f:
        json.dump(zs_preds, f)

    # ── fine-tuning ────────────────────────────────────────────────
    print("Fine-tuning ...")
    del model
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=8192, load_in_4bit=False, token=HF_TOKEN)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )

    texts = [
        tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False)
        for item in train_data
    ]

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir=os.path.join(RESULTS_DIR, f"ft_{level}"),
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            save_strategy="no",
            dataset_text_field="text",
            max_seq_length=8192,
        ),
    )
    trainer.train()

    # ── fine-tuned inference ───────────────────────────────────────
    print("Fine-tuned inference ...")
    FastLanguageModel.for_inference(model)
    ft_preds = run_inference(model, tokenizer, test_data, TASK, "FT")

    if TASK == "ranking":
        ft_m = compute_ranking_metrics(ft_preds)
        print(f"  FT: Acc@1={ft_m['acc_at_1']:.4f}  "
              f"Acc@5={ft_m['acc_at_5']:.4f}  MRR={ft_m['mrr']:.4f}")
    else:
        ft_m = compute_duration_metrics(ft_preds)
        print(f"  FT: MAE={ft_m['mae']:.4f}  RMSE={ft_m['rmse']:.4f}")

    with open(os.path.join(
            RESULTS_DIR, f"llama_finetuned_{TIER}__{TASK}_{level}.json"), "w") as f:
        json.dump(ft_preds, f)

    del model
    torch.cuda.empty_cache()

print("\nDone!")
