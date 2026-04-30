export PATH=$PATH:/opt/rocm/bin
#!/bin/bash
set -e  # Exit on any error

echo "=== Step 1: Verify ROCm is installed ==="
rocminfo | grep "Agent Type" | head -5
if [ $? -ne 0 ]; then
    echo "ERROR: ROCm not found. Wrong instance type?"
    exit 1
fi

echo "=== Step 2: Create virtual environment ==="
python3 -m venv nyaya_env
source nyaya_env/bin/activate

echo "=== Step 3: Install PyTorch for ROCm ==="
# Check ROCm version first
ROCM_VERSION=$(rocminfo | grep "ROCm" | head -1)
echo "Detected: $ROCM_VERSION"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

echo "=== Step 4: Verify PyTorch sees GPU ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available (ROCm): {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
# STOP if GPU not detected
python -c "import torch; assert torch.cuda.is_available(), 'GPU NOT DETECTED - STOP'" 

echo "=== Step 5: Install Flash Attention 2 ==="
pip install ninja
pip install flash-attn --no-build-isolation
# This takes 20-30 mins, get coffee

echo "=== Step 6: Install training dependencies ==="
pip install \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    trl>=0.8.0 \
    datasets \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    sentencepiece \
    protobuf

echo "=== Step 7: Verify bitsandbytes sees GPU ==="
python -c "
import bitsandbytes as bnb
print(f'bitsandbytes version: {bnb.__version__}')
# Check for the warning about no GPU support
import warnings
warnings.filterwarnings('error')
try:
    import bitsandbytes.cuda_setup.main
    print('bitsandbytes: GPU support confirmed')
except Exception as e:
    print(f'WARNING: {e}')
    print('Try: pip install bitsandbytes --upgrade')
"

echo "=== Step 8: HuggingFace login ==="
echo "You need a HuggingFace token with Llama 3.1 access"
echo "Get token from: https://huggingface.co/settings/tokens"
echo "Request model access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
hf auth login

echo "=== Setup Complete ==="
echo "Now run: python smoke_test.py"
echo "If smoke test passes, run: python finetune_amd.py"
