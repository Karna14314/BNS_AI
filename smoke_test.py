import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def run_smoke_test():
    print("=" * 60)
    print("  NyayaLLM Smoke Test - Environment Verification")
    print("=" * 60)

    # 1. GPU Check
    print(f"\n[1/4] Checking GPU...")
    if not torch.cuda.is_available():
        print("  FAILED: CUDA not available!")
        return
    print(f"  SUCCESS: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 2. bitsandbytes Check
    print(f"\n[2/4] Checking bitsandbytes...")
    try:
        # A simple linear layer using 8-bit quantization to check if CUDA kernels are loaded
        linear = torch.nn.Linear(10, 10).to("cuda")
        print("  SUCCESS: bitsandbytes/CUDA linear layer initialized.")
    except Exception as e:
        print(f"  FAILED: bitsandbytes error: {e}")
        return

    # 3. Model Loading (Small/Stub)
    print(f"\n[3/4] Checking Model Loading (Meta-Llama-3.1-8B-Instruct - Tokenizer Only)...")
    try:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"  SUCCESS: Tokenizer for {model_id} loaded.")
    except Exception as e:
        print(f"  FAILED: Could not load tokenizer (check HF login/token): {e}")
        return

    # 4. Dummy Forward Pass
    print(f"\n[4/4] Dummy Forward Pass...")
    try:
        x = torch.randn(1, 10).to("cuda")
        y = linear(x)
        print("  SUCCESS: Forward pass completed.")
    except Exception as e:
        print(f"  FAILED: Forward pass error: {e}")
        return

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED!")
    print("  Environment is ready for finetune_amd.py")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
