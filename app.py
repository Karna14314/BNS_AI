import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import threading
import os

# ─── Configuration ───────────────────────────────────────────
REPO_ID = "ncncomplete/NyayaLLM-Q4_K_M-GGUF"
FILENAME = "nyayallm-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are NyayaLLM, an expert Indian criminal law assistant "
    "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
    "legal information with specific section references."
)

_llm = None
_model_loading = False

def get_llm():
    """Lazy initialization of the Llama model with memory optimization."""
    global _llm, _model_loading
    if _llm is None:
        _model_loading = True
        print(f"Downloading model from {REPO_ID}...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print("Initializing Llama model (Optimized for 16GB RAM)...")
        _llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,      # Fixed threads for stability on Spaces
            n_gpu_layers=0,    # Force CPU
            n_batch=128,       # Reduced batch size for lower memory peak
            use_mmap=True,     # Memory mapping for disk-based paging
            use_mlock=False,   # Do not pin pages to RAM
            verbose=False,
        )
        _model_loading = False
    return _llm

# ─── Inference ───────────────────────────────────────────────
def format_prompt(message, history):
    """Format history and message into Llama 3.1 template."""
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    
    for item in history:
        # Gradio 4.x passes dicts; older versions pass tuples
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if role and content:
                # Llama 3.1 uses 'user' and 'assistant' roles
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        else:
            # fallback for tuple format (user_msg, assistant_msg)
            try:
                user_msg, assistant_msg = item
                if user_msg:
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                if assistant_msg:
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"
            except (TypeError, ValueError):
                continue
    
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def respond(message: str, history: list):
    """Generator for streaming chat responses."""
    llm = get_llm()
    prompt = format_prompt(message, history)
    
    output = llm(
        prompt,
        max_tokens=512,
        stop=["<|eot_id|>", "User:", "System:"],
        echo=False,
        stream=True,
    )
    
    token_sequence = ""
    for chunk in output:
        token = chunk["choices"][0]["text"]
        token_sequence += token
        yield token_sequence

# ─── Gradio UI ───────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    concurrency_limit=1,
    title="⚖️ NyayaLLM — 2023 Indian Criminal Law Assistant",
    description=(
        "Ask questions about **Bharatiya Nyaya Sanhita (BNS)**, "
        "**Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and "
        "**Bharatiya Sakshya Adhiniyam (BSA) 2023**. "
        "Optimized for 16GB RAM using GGUF memory-mapping."
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
