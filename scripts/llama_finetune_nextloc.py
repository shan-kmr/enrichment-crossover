#!/usr/bin/env python3
"""Llama 3.1 8B fine-tuning + zero-shot eval for next-location prediction.

Run on Bouchet HPC with H200 GPU.  Change TIER below per session.
Each level: zero-shot eval → fine-tune on 20K → fine-tuned eval.

Usage on HPC:
  TIER=latlng   → reads ~/nextloc_data_latlng/,   writes ~/nextloc_results_latlng/
  TIER=venue_id → reads ~/nextloc_data_venue_id/, writes ~/nextloc_results_venue_id/
  TIER=enriched → reads ~/nextloc_data_enriched/, writes ~/nextloc_results_enriched/
"""

import json
import logging
import os
import random
import torch
from collections import defaultdict

logging.disable(logging.WARNING)

from unsloth import FastLanguageModel

# ── CHANGE THIS PER RUN ────────────────────────────────────────────
TIER = "latlng"  # "latlng" | "venue_id" | "enriched"
# ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.expanduser(f"~/nextloc_data_{TIER}")
RESULTS_DIR = os.path.expanduser(f"~/nextloc_results_{TIER}")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LEVELS = ["G0", "G1", "G2", "G3", "G4"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_ranking(predictions):
    user_preds = defaultdict(list)
    for p in predictions:
        s = 1.0 if p["prediction"].strip().lower().startswith("yes") else 0.0
        user_preds[p["user_id"]].append((s, p["label"]))

    acc1, acc5, mrr = [], [], []
    for uid, preds in user_preds.items():
        rng = random.Random(uid)
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


os.makedirs(RESULTS_DIR, exist_ok=True)

for level in LEVELS:
    print(f"\n{'=' * 60}")
    print(f"TIER={TIER}  LEVEL={level}")
    print(f"{'=' * 60}")

    train_data = load_jsonl(os.path.join(DATA_DIR, f"train_{level}.jsonl"))
    test_data = load_jsonl(os.path.join(DATA_DIR, f"test_{level}.jsonl"))
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # ── load base model ────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=4096,
        load_in_4bit=False, token=HF_TOKEN,
    )

    # ── zero-shot inference ────────────────────────────────────────
    print("Zero-shot inference …")
    FastLanguageModel.for_inference(model)
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
            "user_id": item["user_id"],
            "candidate_location_id": item["candidate_location_id"],
            "label": item["label"],
            "prediction": resp,
        })
        if (i + 1) % 500 == 0:
            print(f"  ZS {i + 1}/{len(test_data)}")

    zs_m = compute_ranking(zs_preds)
    print(f"  ZS: Acc@1={zs_m['acc_at_1']:.4f}  "
          f"Acc@5={zs_m['acc_at_5']:.4f}  MRR={zs_m['mrr']:.4f}")
    with open(os.path.join(
            RESULTS_DIR, f"llama_zeroshot_{TIER}__{level}.json"), "w") as f:
        json.dump(zs_preds, f)

    # ── fine-tuning ────────────────────────────────────────────────
    print("Fine-tuning …")
    del model
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=4096,
        load_in_4bit=False, token=HF_TOKEN,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )

    texts = [
        tokenizer.apply_chat_template(
            item["messages"], tokenize=False,
            add_generation_prompt=False)
        for item in train_data
    ]

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
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
            max_seq_length=4096,
        ),
    )
    trainer.train()

    # ── fine-tuned inference ───────────────────────────────────────
    print("Fine-tuned inference …")
    FastLanguageModel.for_inference(model)
    ft_preds = []
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
        ft_preds.append({
            "user_id": item["user_id"],
            "candidate_location_id": item["candidate_location_id"],
            "label": item["label"],
            "prediction": resp,
        })
        if (i + 1) % 500 == 0:
            print(f"  FT {i + 1}/{len(test_data)}")

    ft_m = compute_ranking(ft_preds)
    print(f"  FT: Acc@1={ft_m['acc_at_1']:.4f}  "
          f"Acc@5={ft_m['acc_at_5']:.4f}  MRR={ft_m['mrr']:.4f}")
    with open(os.path.join(
            RESULTS_DIR, f"llama_finetuned_{TIER}__{level}.json"), "w") as f:
        json.dump(ft_preds, f)

    del model
    torch.cuda.empty_cache()

print("\nDone!")
