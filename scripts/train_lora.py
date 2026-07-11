"""LoRA fine-tune of Gemma on escalated traffic. Usage (GPU box):
    python scripts/train_lora.py data/datasets/sft_XXXX.jsonl
Registers the result as a candidate version - promotion happens only
after scripts/eval_gate.py passes it."""

import json
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings
from app.flywheel import registry

BASE_MODEL = "google/gemma-2-2b-it"
EPOCHS = 2
LR = 2e-4
LORA_R = 16


def main(dataset_path: str) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    s = get_settings()
    rows = sum(1 for _ in open(dataset_path))
    print(f"Training on {rows} examples from {dataset_path}")
    with open(dataset_path) as f:
        last_id_probe = None  # dataset carries no ids; registry uses meta file
    meta_path = dataset_path + ".meta"
    up_to_id = json.load(open(meta_path))["up_to_id"] if os.path.exists(meta_path) else 0

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=dataset_path, split="train")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(s.adapters_dir, f"adapter_{stamp}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=LoraConfig(r=LORA_R, lora_alpha=32, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(output_dir=out_dir, num_train_epochs=EPOCHS,
                       per_device_train_batch_size=2, gradient_accumulation_steps=4,
                       learning_rate=LR, bf16=True, logging_steps=5,
                       save_strategy="no", report_to=[]),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)

    v = registry.register_candidate(out_dir, trained_rows=rows, up_to_id=up_to_id)
    print(f"Registered candidate: {v['name']} -> {out_dir}")
    print("Next: python scripts/eval_gate.py", v["name"])


if __name__ == "__main__":
    main(sys.argv[1])