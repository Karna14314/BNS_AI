"""
NyayaLLM Fine-Tuning Script for AMD MI300X (ROCm)
==================================================
Fine-tunes Llama 3.1 8B Instruct on Indian criminal law dataset
(BNS, BNSS, BSA 2023) using QLoRA.

Usage:
    python finetune_amd.py

Requirements:
    pip install torch transformers peft trl datasets bitsandbytes accelerate
"""

# ============================================================
# Environment Setup — MUST be before any torch/HIP imports
# ============================================================
import os
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================
# Imports
# ============================================================
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import json
import random

# ============================================================
# Constants
# ============================================================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE = "final_training_data.jsonl"
OUTPUT_DIR = "./nyaya-llama-output"
FINAL_DIR = "./nyaya-llama-final"
EVAL_SET_FILE = "eval_set.jsonl"
MAX_SEQ_LENGTH = 2048
SEED = 42


# ============================================================
# Step 1: Load Dataset from Hugging Face
# ============================================================
def load_and_split_data():
    """Load from HF Hub, shuffle, split 40 for eval, and save eval set."""
    print("=" * 60)
    print("  Step 1: Loading dataset from Hugging Face Hub")
    print("=" * 60)
    
    # Load from the repo pushed earlier
    dataset = load_dataset("ncncomplete/NyayaLLM-Dataset", split="train")
    
    # Shuffle and split 40 samples for evaluation
    dataset = dataset.shuffle(seed=SEED)
    split_dataset = dataset.train_test_split(test_size=40, seed=SEED)
    
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Save eval set for the evaluation script (evaluate.py expects this)
    import json
    with open(EVAL_SET_FILE, "w", encoding="utf-8") as f:
        for sample in eval_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Saved {len(eval_dataset)} eval samples to {EVAL_SET_FILE}")

    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


# ============================================================
# Step 2: Format Function (Llama 3.1 Instruct Chat Template)
# ============================================================
def format_sample(sample):
    """Convert Alpaca-format sample to Llama 3.1 chat template."""
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    output = sample["output"]

    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    # Llama 3.1 Instruct chat format
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are NyayaLLM, an expert Indian criminal law assistant "
        "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
        "legal information with specific section references."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}"
        "<|eot_id|>"
    )
    return {"text": text}


# ============================================================
# Step 3: Quantization Config (QLoRA — 4-bit NF4)
# ============================================================
def get_bnb_config():
    """Create BitsAndBytes 4-bit quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4GB
    )


# ============================================================
# Step 4: Load Model and Tokenizer
# ============================================================
def load_model_and_tokenizer(bnb_config):
    """Load Llama 3.1 8B with 4-bit quantization."""
    print("\n" + "=" * 60)
    print("  Step 4: Loading model and tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # flash_attention_2 if ROCm supports it
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Print model memory footprint
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {MODEL_ID}")
    print(f"  Total params: {total_params:,}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return model, tokenizer


# ============================================================
# Step 5: LoRA Config
# ============================================================
def get_lora_config():
    """Create LoRA configuration targeting all linear layers."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )


# ============================================================
# Step 6: Training Arguments
# ============================================================
def get_training_args():
    """Create training arguments optimized for MI300X."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=4,
        group_by_length=True,
        report_to="none",
        seed=SEED,
    )


# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  NyayaLLM Fine-Tuning Pipeline (AMD MI300X / ROCm)")
    print("=" * 60)

    # Verify GPU
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("  [WARN] No GPU detected! Training will be extremely slow on CPU.")

    # Step 1: Data
    train_dataset, eval_dataset = load_and_split_data()

    # Step 2: Format
    print("\n  Formatting datasets with Llama 3.1 chat template...")
    train_dataset = train_dataset.map(format_sample)
    eval_dataset = eval_dataset.map(format_sample)

    # Step 3: Quantization
    bnb_config = get_bnb_config()

    # Step 4: Model + Tokenizer
    model, tokenizer = load_model_and_tokenizer(bnb_config)

    # Step 5: LoRA
    lora_config = get_lora_config()

    # Print LoRA trainable params
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"\n  LoRA trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # Step 6: Training args
    training_args = get_training_args()

    # Step 7: Trainer
    print("\n" + "=" * 60)
    print("  Step 7: Initializing SFTTrainer")
    print("=" * 60)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        packing=True,  # CRITICAL for GPU utilization
    )

    # Step 8: Train
    print("\n" + "=" * 60)
    print("  Step 8: Starting Training")
    print("=" * 60)
    print(f"  Train samples:        {len(train_dataset)}")
    print(f"  Eval samples:         {len(eval_dataset)}")
    print(f"  Batch size:           4")
    print(f"  Gradient accum steps: 4")
    print(f"  Effective batch size: {4 * 4} = 16")
    print(f"  Epochs:               3")
    print(f"  Max sequence length:  {MAX_SEQ_LENGTH}")
    print(f"  Learning rate:        2e-4")
    print(f"  LoRA rank:            16")
    print()

    train_result = trainer.train()

    # Print training metrics
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Training loss:  {train_result.training_loss:.4f}")

    # Final eval
    eval_result = trainer.evaluate()
    print(f"  Final eval loss: {eval_result['eval_loss']:.4f}")

    # Step 9: Save
    print("\n" + "=" * 60)
    print("  Step 9: Saving Model")
    print("=" * 60)

    # Save LoRA adapter
    trainer.save_model(OUTPUT_DIR)
    print(f"  LoRA adapter saved to {OUTPUT_DIR}")

    # Merge LoRA weights into base model for deployment
    print("  Merging LoRA weights into base model...")
    from peft import AutoPeftModelForCausalLM
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print(f"  Merged model saved to {FINAL_DIR}")
    print()
    print("=" * 60)
    print("  DONE! Next steps:")
    print("=" * 60)
    print(f"  1. Evaluate: python scripts/evaluate.py")
    print(f"  2. Upload:   huggingface-cli upload YOUR_HF_USERNAME/NyayaLLM {FINAL_DIR}")
    print()


if __name__ == "__main__":
    main()
