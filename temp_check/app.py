import gradio as gr
from huggingface_hub import InferenceClient

# ─── Configuration ───────────────────────────────────────────
MODEL_ID = "ncncomplete/NyayaLLM"
SYSTEM_PROMPT = (
    "You are NyayaLLM, an expert Indian criminal law assistant "
    "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
    "legal information with specific section references."
)

import os
# Use HF Inference API — no local GPU needed
# Note: You must add your HF_TOKEN as a Secret in the Space settings
client = InferenceClient(model=MODEL_ID, token=os.environ.get("HF_TOKEN"))

# ─── Inference ───────────────────────────────────────────────
def respond(message, history):
    """History is list of (user, assistant) tuples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ─── Gradio UI ───────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="⚖️ NyayaLLM — 2023 Indian Criminal Law Assistant",
    description=(
        "Ask questions about **Bharatiya Nyaya Sanhita (BNS)**, "
        "**Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and "
        "**Bharatiya Sakshya Adhiniyam (BSA) 2023**. "
        "Fine-tuned on AMD MI300X · 99.2% evaluation accuracy."
    ),
    examples=[
        "What is the punishment for murder under BNS 2023?",
        "How has the definition of sedition changed compared to IPC Section 124A?",
        "What is the procedure for a zero FIR under BNSS?",
        "Under BSA, what is the evidentiary value of electronic records?",
        "Which section of BNSS covers anticipatory bail?",
    ],
)

if __name__ == "__main__":
    demo.launch()
