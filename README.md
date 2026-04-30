---
title: NyayaLLM
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
license: llama3.1
---

# NyayaLLM: Indian Criminal Law (BNS/BNSS/BSA) Fine-Tuning

This is the official Space for **NyayaLLM**, a fine-tuned Llama 3.1 8B model specialized in the new 2023 Indian Criminal Laws. 

## Model Details
- **Model Name**: NyayaLLM
- **Format**: GGUF (Q4_K_M)
- **Infrastructure**: Fine-tuned on AMD MI300X.
- **Evaluation Accuracy**: 99.2%

## Project Structure
- `app.py`: Main Gradio application using `llama-cpp-python`.
- `requirements.txt`: Minimal dependencies for GGUF inference.
- `scripts/`: Training and evaluation scripts.
- `data/`: Processed datasets and document chunks.
- `results/`: Benchmarking and evaluation outputs.
