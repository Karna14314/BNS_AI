# NyayaLLM: Indian Criminal Law (BNS/BNSS/BSA) Fine-Tuning

This repository contains a high-fidelity pipeline for extracting, generating, and fine-tuning LLMs on the new Indian Criminal Laws (Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, and Bharatiya Sakshya Adhiniyam 2023).

## Project Structure

The project has been reorganized for clarity:

- **Root**: Core training scripts and final dataset.
  - `finetune_amd.py`: Main QLoRA fine-tuning script optimized for AMD MI300X.
  - `final_training_data.jsonl`: The curated dataset used for training.
  - `requirements.txt`: Python dependencies.
- **`data/`**: Managed data storage.
  - `data/processed/`: Intermediate JSONL and CSV datasets.
  - `data/chunks/`: Document chunks for processing.
  - `data/md/`: Extracted markdown from PDFs.
- **`scripts/`**: Utility and pipeline scripts.
  - `generate_dataset.py`: AI-driven scenario generation.
  - `evaluate.py`: Performance benchmark script.
  - `extract_and_chunk.py`: PDF to Chunk pipeline.
- **`raw_docs/`**: Source PDF documents (BNS, BNSS, BSA).
- **`logs/`**: Training and processing logs.

## Setup
Ensure you have the required dependencies and your `.env` configured with the NVIDIA API key (for generation) or HuggingFace token.

```bash
pip install -r requirements.txt
```

## Dataset Generation
To generate the dataset from scratch using the raw PDFs:
1. Extract and chunk: `python scripts/extract_and_chunk.py`
2. Generate scenarios: `python scripts/generate_dataset.py`

## Fine-Tuning
To start the fine-tuning process on an AMD GPU:
```bash
python finetune_amd.py
```

## Evaluation
After fine-tuning, evaluate the model performance:
```bash
python scripts/evaluate.py
```
