import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# ─── Configuration ───────────────────────────────────────────
REPO_ID = "ncncomplete/NyayaLLM-Q4_K_M-GGUF"
FILENAME = "nyayallm-q4_k_m.gguf"

# Load the model and tokenizer
print(f"Loading GGUF model from {REPO_ID}...")
# We use AutoModelForCausalLM to load GGUF directly
model = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    gguf_file=FILENAME,
    torch_dtype=torch.float32, # CPU friendly
    device_map="cpu"
)

# For GGUF models, we often need the base tokenizer if not in the GGUF
# Llama 3.1 base tokenizer is usually compatible
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

SYSTEM_PROMPT = (
    "You are NyayaLLM, an expert Indian criminal law assistant "
    "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
    "legal information with specific section references."
)

# ─── Inference ───────────────────────────────────────────────
def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Format using Llama 3.1 template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# ─── Gradio UI ───────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="⚖️ NyayaLLM — 2023 Indian Criminal Law Assistant",
    description=(
        "Ask questions about **Bharatiya Nyaya Sanhita (BNS)**, "
        "**Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and "
        "**Bharatiya Sakshya Adhiniyam (BSA) 2023**. "
        "Running on GGUF (Q4_K_M) via Transformers native support."
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
