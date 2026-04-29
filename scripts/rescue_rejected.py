import json
import os
import re
from collections import Counter, defaultdict

# Configuration
REJECTED_FILE = "rejected_scenarios.jsonl"
FINAL_FILE = "training_data_final.jsonl"
RESCUED_OUTPUT = "rescued_samples.jsonl"

def rescue_samples():
    if not os.path.exists(REJECTED_FILE):
        print(f"Error: {REJECTED_FILE} not found.")
        return

    rejected_data = []
    with open(REJECTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rejected_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_rejected = len(rejected_data)
    rescued_samples = []
    reason_stats = Counter()

    # Regex patterns
    section_patterns = [
        r'Section\s+\d+[\(\w\)]*\s*(of\s+)?(BNS|BNSS|BSA|Bharatiya)',
        r'(BNS|BNSS|BSA)\s+Section\s+\d+',
        r'Section\s+\d+[\(\w\)]*\s+of\s+(the\s+)?(Bharatiya|Sanhita|Adhiniyam)'
    ]
    
    # Phrases to avoid
    bad_phrases = ["not specified in source", "cannot be determined", "section not specified"]

    for entry in rejected_data:
        reason = entry.get("validation_reason", "unknown")
        instruction = entry.get("instruction", "")
        output = entry.get("output", "")
        combined_text = (instruction + " " + output)
        combined_lower = combined_text.lower()
        
        words = re.findall(r'\b\w+\b', output)
        if len(words) < 40:
            continue

        # Skip major red flags
        if any(r in reason for r in ["output_too_short", "model_flagged_bad_chunk"]):
            continue
        if any(f in reason for f in ["Companies Act", "Evidence Act"]):
            continue

        is_rescued = False
        rescue_reason = ""

        # Case 1: missing_keys: {'input'}
        if "missing_keys" in reason and "'input'" in reason:
            if "input" not in entry:
                entry["input"] = ""
            is_rescued = True
            rescue_reason = "missing_input_key"

        # Case 2: no_valid_section_reference
        elif reason == "no_valid_section_reference":
            # Must have pattern
            has_pattern = any(re.search(p, combined_text, re.IGNORECASE) for p in section_patterns)
            # Must NOT have bad phrases
            has_bad_phrase = any(bp in combined_lower for bp in bad_phrases)
            
            if has_pattern and not has_bad_phrase:
                is_rescued = True
                rescue_reason = "recovered_section_ref"

        # Case 3: forbidden_reference: CrPC or IPC (comparative)
        elif "forbidden_reference" in reason:
            is_comparative = False
            target_law = ""
            if "CrPC" in reason:
                target_law = "crpc"
            elif "Indian Penal Code" in reason or "IPC" in reason:
                target_law = "ipc"
            
            if target_law:
                # Check for comparative language
                comparative_phrases = ["corresponds to", "replacing", "equivalent to", "replaces", "similar to", "compared to"]
                if any(cp in combined_lower for cp in comparative_phrases):
                    # Count mentions
                    bnss_mentions = len(re.findall(r'bnss|bns|bsa|bharatiya', combined_lower))
                    forbidden_mentions = len(re.findall(re.escape(target_law), combined_lower))
                    
                    if forbidden_mentions > 0 and (bnss_mentions / forbidden_mentions) > 2:
                        is_rescued = True
                        rescue_reason = f"comparative_{target_law}"

        if is_rescued:
            # Prepare for output
            rescued_entry = {
                "instruction": entry.get("instruction", ""),
                "input": entry.get("input", ""),
                "output": entry.get("output", ""),
            }
            rescued_samples.append(rescued_entry)
            reason_stats[reason] += 1

    # Save to files
    if rescued_samples:
        # Save to rescued_samples.jsonl
        with open(RESCUED_OUTPUT, "w", encoding="utf-8") as f:
            for s in rescued_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        # Append to training_data_final.jsonl
        with open(FINAL_FILE, "a", encoding="utf-8") as f:
            for s in rescued_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*40}")
    print(f" RESCUE MISSION REPORT")
    print(f"{'='*40}")
    print(f"Total rejected   : {total_rejected}")
    print(f"Rescued samples  : {len(rescued_samples)}")
    print(f"Still rejected   : {total_rejected - len(rescued_samples)}")
    
    if rescued_samples:
        print("\nRescue breakdown by original reason:")
        for r, count in reason_stats.items():
            print(f"  - {r}: {count}")
    
    print(f"\nRescued samples appended to: {FINAL_FILE}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    rescue_samples()
