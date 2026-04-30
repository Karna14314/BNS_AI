"""
NyayaLLM Evaluation Script
===========================
Compares base Llama 3.1 8B vs fine-tuned NyayaLLM on:
  1. Held-out eval set (40 samples from training split)
  2. 40 hardcoded benchmark questions covering BNS/BNSS/BSA

Usage:
    python evaluate.py
"""

import os
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Constants
# ============================================================
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
FINETUNED_DIR = "./nyaya-llama-final"
EVAL_SET_FILE = "eval_set.jsonl"
RESULTS_FILE = "evaluation_results.json"
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are NyayaLLM, an expert Indian criminal law assistant "
    "specializing in BNS, BNSS, and BSA 2023. Provide accurate "
    "legal information with specific section references."
)


# ============================================================
# 40 Hardcoded Benchmark Questions
# ============================================================
BENCHMARK_QUESTIONS = [
    # --- 10 BNS Section Identification ---
    {
        "instruction": "A man attacks his neighbor with a knife causing grievous hurt. Which BNS section applies and what is the punishment?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A woman is kidnapped from her home for the purpose of forced marriage. Under which section of BNS is this offence defined?",
        "category": "bns_section_id"
    },
    {
        "instruction": "An employee embezzles company funds worth 10 lakh rupees. Which BNS section covers criminal breach of trust and what is the punishment?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A mob of seven people lynches a man based on his religion. Which BNS section specifically addresses mob lynching?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A person sends threatening messages to a woman on social media demanding sexual favours. Which BNS section applies to this cyber-stalking offence?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A doctor performs an operation negligently and the patient dies. Which section of BNS deals with death caused by negligence by a medical professional?",
        "category": "bns_section_id"
    },
    {
        "instruction": "Two groups engage in a street fight and one person dies. Which BNS section defines the offence of culpable homicide not amounting to murder?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A person forges a property deed to sell land that does not belong to him. Which section of BNS covers forgery and what punishment does it carry?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A husband subjects his wife to mental and physical cruelty. Under which section of BNS can she file a complaint for cruelty by husband?",
        "category": "bns_section_id"
    },
    {
        "instruction": "A public servant demands a bribe of 50,000 rupees for issuing a license. Which section of BNS defines the offence of bribery by a public servant?",
        "category": "bns_section_id"
    },

    # --- 10 BNSS Procedure ---
    {
        "instruction": "A person is arrested without a warrant. What are the mandatory procedures the police must follow under BNSS at the time of arrest?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "Under BNSS, what is the maximum period for which police custody remand can be granted by a Magistrate?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "An FIR has been filed for a cognizable offence. Under which section of BNSS is the procedure for FIR registration defined?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "The accused wants to apply for anticipatory bail. Under which section of BNSS can anticipatory bail be granted and what conditions can the court impose?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "A witness refuses to appear before the court. What powers does the Magistrate have under BNSS to compel the attendance of witnesses?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "Under BNSS, what is the procedure for conducting a search of a premises without a warrant? When is this permitted?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "The prosecution wants to withdraw a criminal case. Under which section of BNSS can the Public Prosecutor withdraw from prosecution?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "An accused person has been in jail for half the maximum sentence period without trial being completed. What are their rights under BNSS regarding default bail?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "Under BNSS, what is the mandatory timeline for completing investigation after filing of FIR for offences punishable with imprisonment of 10 years or more?",
        "category": "bnss_procedure"
    },
    {
        "instruction": "A victim of sexual assault wants to record her statement. Under which section of BNSS must her statement be recorded by a Magistrate, and what protections are available?",
        "category": "bnss_procedure"
    },

    # --- 8 BSA Evidence ---
    {
        "instruction": "What are the rules under BSA for admissibility of electronic evidence such as WhatsApp messages and emails in court?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "Under BSA, when can a confession made to the police be considered admissible in court proceedings?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "A dying declaration is made by a murder victim to a bystander. Under which section of BSA is a dying declaration admissible as evidence?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "Under BSA, what is the evidentiary value of DNA evidence in criminal trials? Which section governs expert opinion evidence?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "The prosecution wants to present CCTV footage as evidence. Under BSA, what conditions must be met for digital video evidence to be admissible?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "Under BSA, what is the doctrine of burden of proof in criminal cases? On whom does the burden lie and when does it shift?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "A character witness is called to testify about the accused's reputation. Under which section of BSA is character evidence admissible in criminal proceedings?",
        "category": "bsa_evidence"
    },
    {
        "instruction": "Under BSA, can a document be proved without calling its author as a witness? What are the rules for proving documents through secondary evidence?",
        "category": "bsa_evidence"
    },

    # --- 7 IPC to BNS Transition ---
    {
        "instruction": "Section 302 of the Indian Penal Code dealt with murder. What is the corresponding section in BNS and has the punishment changed?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "IPC Section 420 covered cheating and dishonestly inducing delivery of property. What is the equivalent section in BNS 2023?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "What new offences have been introduced in BNS 2023 that did not exist in the old Indian Penal Code?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "IPC Section 376 dealt with rape. How has the corresponding BNS section changed the definition and punishment for sexual assault offences?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "Section 498A IPC dealt with cruelty by husband. What is the equivalent provision in BNS and are there any changes?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "IPC Section 304A covered death by negligence. How does the corresponding BNS section treat negligent death differently, especially for hit-and-run cases?",
        "category": "ipc_to_bns"
    },
    {
        "instruction": "How does BNS 2023 handle sedition compared to the old IPC Section 124A? Has sedition been retained, modified, or removed?",
        "category": "ipc_to_bns"
    },

    # --- 5 Punishment ---
    {
        "instruction": "What is the maximum punishment for murder under BNS? In which cases can the death penalty be imposed?",
        "category": "punishment"
    },
    {
        "instruction": "What is the punishment for sexual harassment in the workplace under BNS? Differentiate between different grades of the offence.",
        "category": "punishment"
    },
    {
        "instruction": "Under BNS, what is the punishment for a hit-and-run accident where the driver causes death and flees without reporting to the police?",
        "category": "punishment"
    },
    {
        "instruction": "What is the punishment for organized crime under BNS 2023? How does BNS define organized crime?",
        "category": "punishment"
    },
    {
        "instruction": "What is the punishment for making a false complaint or giving false information to the police under BNS?",
        "category": "punishment"
    },
]


# ============================================================
# Scoring Function
# ============================================================
def score_output(output_text):
    """Score a model output 0-3 based on quality criteria.

    1 point: contains a section number (e.g. Section 103)
    1 point: mentions BNS/BNSS/BSA specifically
    1 point: output > 50 words (specific, not vague)
    """
    score = 0
    reasons = []

    # Check for section number
    if re.search(r'Section\s+\d+', output_text, re.IGNORECASE):
        score += 1
        reasons.append("has_section_number")

    # Check for BNS/BNSS/BSA mention
    if re.search(r'\b(BNS|BNSS|BSA|Bharatiya\s+Nyaya|Bharatiya\s+Nagarik|Bharatiya\s+Sakshya)\b',
                  output_text, re.IGNORECASE):
        score += 1
        reasons.append("mentions_new_law")

    # Check output length
    word_count = len(re.findall(r'\b\w+\b', output_text))
    if word_count > 50:
        score += 1
        reasons.append(f"detailed ({word_count} words)")

    return score, reasons


# ============================================================
# Inference Function
# ============================================================
def generate_response(model, tokenizer, question, device):
    """Generate a response using the Llama 3.1 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (exclude the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ============================================================
# Load Model Helper
# ============================================================
def load_model(model_path, is_quantized=False):
    """Load a model and tokenizer."""
    print(f"  Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"  Loaded on device: {device}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer, device


# ============================================================
# Main Evaluation Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("  NyayaLLM Evaluation: Base vs Fine-Tuned")
    print("=" * 70)

    # --- Load eval set ---
    eval_samples = []
    if os.path.exists(EVAL_SET_FILE):
        with open(EVAL_SET_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        eval_samples.append({
                            "instruction": sample["instruction"],
                            "category": "eval_holdout",
                            "reference_output": sample.get("output", ""),
                        })
                    except json.JSONDecodeError:
                        continue
        print(f"  Loaded {len(eval_samples)} held-out eval samples")
    else:
        print(f"  [WARN] {EVAL_SET_FILE} not found. Skipping held-out eval.")

    # --- Combine with benchmark questions ---
    benchmark_items = []
    for q in BENCHMARK_QUESTIONS:
        benchmark_items.append({
            "instruction": q["instruction"],
            "category": q["category"],
            "reference_output": "",
        })

    all_questions = eval_samples + benchmark_items
    print(f"  Total questions: {len(all_questions)} "
          f"({len(eval_samples)} eval + {len(benchmark_items)} benchmark)")

    # --- Load base model ---
    print("\n" + "-" * 70)
    print("  Loading BASE model...")
    print("-" * 70)
    base_model, base_tokenizer, base_device = load_model(BASE_MODEL_ID)

    # --- Run base model inference ---
    print("\n  Running base model inference...")
    base_results = []
    for i, q in enumerate(all_questions):
        print(f"    [{i+1}/{len(all_questions)}] {q['instruction'][:60]}...")
        response = generate_response(base_model, base_tokenizer, q["instruction"], base_device)
        score, reasons = score_output(response)
        base_results.append({
            "response": response,
            "score": score,
            "reasons": reasons,
        })

    # Free base model memory
    del base_model
    torch.cuda.empty_cache()
    print("  Base model unloaded.")

    # --- Load fine-tuned model ---
    ft_results = []
    ft_loaded = False
    
    if os.path.exists(FINETUNED_DIR):
        print("\n" + "-" * 70)
        print("  Loading FINE-TUNED model...")
        print("-" * 70)
        try:
            ft_model, ft_tokenizer, ft_device = load_model(FINETUNED_DIR)
            
            # --- Run fine-tuned model inference ---
            print("\n  Running fine-tuned model inference...")
            for i, q in enumerate(all_questions):
                print(f"    [{i+1}/{len(all_questions)}] {q['instruction'][:60]}...")
                response = generate_response(ft_model, ft_tokenizer, q["instruction"], ft_device)
                score, reasons = score_output(response)
                ft_results.append({
                    "response": response,
                    "score": score,
                    "reasons": reasons,
                })
            
            del ft_model
            torch.cuda.empty_cache()
            ft_loaded = True
        except Exception as e:
            print(f"  [ERROR] Failed to load fine-tuned model: {e}")
    else:
        print(f"\n  [INFO] Fine-tuned model not found at {FINETUNED_DIR}. Proceeding with Baseline only.")
        # Fill ft_results with dummy data for the table
        for _ in all_questions:
            ft_results.append({"response": "N/A", "score": 0, "reasons": []})

    # ============================================================
    # Print Side-by-Side Comparison Table
    # ============================================================
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    # Category-level stats
    category_stats = {}
    for i, q in enumerate(all_questions):
        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = {"base_total": 0, "ft_total": 0, "count": 0}
        category_stats[cat]["base_total"] += base_results[i]["score"]
        category_stats[cat]["ft_total"] += ft_results[i]["score"]
        category_stats[cat]["count"] += 1

    print(f"\n  {'Category':<25} {'Base Avg':>10} {'FT Avg':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    for cat, stats in sorted(category_stats.items()):
        base_avg = stats["base_total"] / stats["count"]
        ft_avg = stats["ft_total"] / stats["count"]
        delta = ft_avg - base_avg
        sign = "+" if delta > 0 else ""
        print(f"  {cat:<25} {base_avg:>10.2f} {ft_avg:>10.2f} {sign}{delta:>9.2f}")

    # Overall
    total_base = sum(r["score"] for r in base_results)
    total_ft = sum(r["score"] for r in ft_results)
    max_possible = len(all_questions) * 3
    print(f"\n  {'OVERALL':<25} {total_base:>10} {total_ft:>10} "
          f"{'+' if total_ft > total_base else ''}{total_ft - total_base:>9}")
    print(f"  {'Max possible':<25} {max_possible:>10} {max_possible:>10}")
    print(f"  {'Score %':<25} {100*total_base/max_possible:>9.1f}% "
          f"{100*total_ft/max_possible:>9.1f}%")

    # Print detailed per-question comparison (first 10)
    print("\n" + "-" * 70)
    print("  SAMPLE COMPARISONS (first 10)")
    print("-" * 70)

    for i in range(min(10, len(all_questions))):
        q = all_questions[i]
        print(f"\n  Q{i+1} [{q['category']}]: {q['instruction'][:80]}...")
        print(f"  BASE  (score {base_results[i]['score']}/3): "
              f"{base_results[i]['response'][:150]}...")
        print(f"  FT    (score {ft_results[i]['score']}/3): "
              f"{ft_results[i]['response'][:150]}...")

    # ============================================================
    # Save Full Results
    # ============================================================
    full_results = {
        "summary": {
            "total_questions": len(all_questions),
            "base_total_score": total_base,
            "ft_total_score": total_ft,
            "max_possible_score": max_possible,
            "base_pct": round(100 * total_base / max_possible, 1),
            "ft_pct": round(100 * total_ft / max_possible, 1),
            "improvement_pct": round(100 * (total_ft - total_base) / max(total_base, 1), 1),
        },
        "category_breakdown": {},
        "detailed_results": [],
    }

    for cat, stats in category_stats.items():
        full_results["category_breakdown"][cat] = {
            "count": stats["count"],
            "base_avg": round(stats["base_total"] / stats["count"], 2),
            "ft_avg": round(stats["ft_total"] / stats["count"], 2),
        }

    for i, q in enumerate(all_questions):
        full_results["detailed_results"].append({
            "question": q["instruction"],
            "category": q["category"],
            "reference_output": q.get("reference_output", ""),
            "base_response": base_results[i]["response"],
            "base_score": base_results[i]["score"],
            "base_reasons": base_results[i]["reasons"],
            "ft_response": ft_results[i]["response"],
            "ft_score": ft_results[i]["score"],
            "ft_reasons": ft_results[i]["reasons"],
        })

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Full results saved to {RESULTS_FILE}")
    print("=" * 70)
    print("  Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
