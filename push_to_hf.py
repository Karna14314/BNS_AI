from datasets import load_dataset
import os

# Define the file path
data_file = "final_training_data.jsonl"
repo_id = "ncncomplete/NyayaLLM-Dataset"

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found!")
    exit(1)

print(f"Loading {data_file}...")
dataset = load_dataset("json", data_files=data_file, split="train")

print(f"Pushing to Hugging Face Hub: {repo_id}...")
dataset.push_to_hub(repo_id, private=True)

print("Done!")
