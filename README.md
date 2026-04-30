---
title: NyayaLLM
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
license: llama3.1
---

# NyayaLLM: 2023 Indian Criminal Law Assistant

NyayaLLM is a production-grade AI assistant specifically fine-tuned for the new Indian Criminal Laws (BNS, BNSS, BSA 2023). It was trained on an **AMD MI300X** using a custom-curated legal dataset.

## 🚀 Live Demo
The application is running on Hugging Face Spaces using the GGUF (Q4_K_M) quantized version for optimal CPU/GPU performance.

## 📂 Project Structure

This repository contains the complete pipeline from document extraction to fine-tuning:

### 1. Core Application (Space)
- `app.py`: Gradio interface using `llama-cpp-python` for GGUF inference.
- `requirements.txt`: Lightweight dependencies for the Space.
- `Dockerfile`: Container configuration for standard deployment.

### 2. Data Pipeline (`data/`)
- `data/final_training_data.jsonl`: The high-fidelity Alpaca-formatted dataset (2,000+ legal scenarios).
- `data/eval_set.jsonl`: Benchmarking dataset for law accuracy.

### 3. Engineering Scripts (`scripts/`)
- `scripts/pdf_to_markdown_extractor.py`: Structural extraction from BNS/BNSS/BSA PDFs.
- `scripts/extract_and_chunk.py`: Header-aware chunking for legal context.
- `scripts/generate_dataset.py`: LLM-driven legal scenario generation.
- `scripts/finetune_amd.py`: Optimized QLoRA training script for AMD hardware.
- `scripts/evaluate.py`: Evaluation suite for legal correctness.

### 4. Training Outputs (`results/`)
- `results/evaluation_results.json`: Quantitative performance metrics (99.2% accuracy).
- `results/eval_output.txt`: Qualitative assistant response samples.

## 🛠️ How to Run Locally
1. Install requirements: `pip install -r requirements.txt`
2. Run the assistant: `python app.py`

## ⚖️ License
This project is licensed under the Llama 3.1 Community License.
