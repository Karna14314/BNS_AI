import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

# ─── Configuration ───────────────────────────────────────────
REPO_ID = "ncncomplete/NyayaLLM-Q4_K_M-GGUF"
FILENAME = "nyayallm-q4_k_m.gguf"

# Download the model
print(f"Downloading model from {REPO_ID}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# Initialize the model
print("Initializing Llama model (CPU Optimized)...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=os.cpu_count(),
    verbose=False
)

SYSTEM_PROMPT = (
    "You are NyayaLLM, an expert Indian criminal law assistant "
    "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
    "legal information with specific section references."
)

# ─── Inference ───────────────────────────────────────────────
def format_prompt(message, history):
    """Format history and message into Llama 3.1 template."""
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    for user_msg, assistant_msg in history:
        if user_msg:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        if assistant_msg:
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def respond(message, history):
    prompt = format_prompt(message, history)
    
    # Use streaming generation
    output = llm(
        prompt,
        max_tokens=512,
        stop=["<|eot_id|>", "User:", "System:"],
        echo=False,
        stream=True
    )
    
    token_sequence = ""
    for chunk in output:
        token = chunk['choices'][0]['text']
        token_sequence += token
        yield token_sequence

# ─── Gradio UI ───────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="⚖️ NyayaLLM — 2023 Indian Criminal Law Assistant",
    description=(
        "Ask questions about **Bharatiya Nyaya Sanhita (BNS)**, "
        "**Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and "
        "**Bharatiya Sakshya Adhiniyam (BSA) 2023**. "
        "Running on GGUF (Q4_K_M) via llama-cpp-python (Pre-built CPU Wheel)."
    ),
    examples=[
        "What is the punishment for murder under BNS 2023?",
        "How has the definition of sedition changed compared to IPC Section 124A?",
        "What is the procedure for a zero FIR under BNSS?",
        "Under BSA, what is the evidentiary value of electronic records?",
        "Which section of BNSS covers anticipatory bail?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
