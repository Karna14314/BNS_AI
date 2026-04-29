import json
import os
import random
import re
import pandas as pd

# Files
CLEAN_DATA_FILE = "training_data_final.jsonl"
REASONING_FILE = "bns_reasoning_dataset.jsonl"
TRANSITION_FILE = "bns_transition_dataset.jsonl"
CSV_FILE = "IPC_and_BNS_transformation.csv"
OUTPUT_FILE = "final_training_data.jsonl"
SOURCES_BREAKDOWN_FILE = "sources_breakdown.json"

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def get_words(text):
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    return set(w for w in words if w not in STOPWORDS)

def load_jsonl(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return data
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def main():
    print(f"Loading data sources...")
    all_data = []
    source_counts = {
        "generated_clean": 0,
        "reasoning": 0,
        "transition": 0,
        "csv_converted": 0,
        "duplicates_removed": 0
    }
    
    # Track the source of each sample
    source_tracking = []

    # STEP 1: Load existing clean data
    clean_data = load_jsonl(CLEAN_DATA_FILE)
    for sample in clean_data:
        all_data.append(sample)
        source_tracking.append("generated_clean")
        source_counts["generated_clean"] += 1

    # STEP 2: Load reasoning dataset
    reasoning_data = load_jsonl(REASONING_FILE)
    for sample in reasoning_data:
        output = sample.get("output", "")
        if "not specified in source" not in output.lower():
            all_data.append(sample)
            source_tracking.append("reasoning")
            source_counts["reasoning"] += 1

    # STEP 3: Load transition dataset
    transition_data = load_jsonl(TRANSITION_FILE)
    for sample in transition_data:
        all_data.append(sample)
        source_tracking.append("transition")
        source_counts["transition"] += 1

    # STEP 4: Convert CSV
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            for idx, row in df.iterrows():
                prompt = str(row.get("prompts", ""))
                response_str = str(row.get("response", ""))
                try:
                    import ast
                    parsed = ast.literal_eval(response_str)
                    bns_section = str(parsed.get("BNS Section", "")).strip()
                    if not bns_section or bns_section.upper() == "N/A":
                        continue
                    
                    instruction = prompt
                    output = (
                        f"IPC Section {parsed.get('IPC Section', '')} ('{parsed.get('IPC Heading', '')}') "
                        f"corresponds to BNS Section {bns_section} "
                        f"('{parsed.get('BNS Heading', '')}').\n\n"
                        f"IPC Text: {str(parsed.get('IPC Descriptions', ''))[:300]}\n\n"
                        f"BNS Text: {str(parsed.get('BNS description', ''))[:400]}"
                    )
                    
                    sample = {
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    }
                    all_data.append(sample)
                    source_tracking.append("csv_converted")
                    source_counts["csv_converted"] += 1
                except (ValueError, SyntaxError):
                    continue
        except Exception as e:
            print(f"Error processing CSV: {e}")
    else:
        print(f"Warning: {CSV_FILE} not found.")

    # STEP 5: Deduplication
    print(f"Total samples before deduplication: {len(all_data)}")
    print(f"Running deduplication...")
    
    unique_data = []
    unique_sources = []
    
    seen_instructions = set() # Exact matches
    processed_word_sets = [] # For near-duplicates
    
    duplicates_removed = 0

    for i, sample in enumerate(all_data):
        instruction = sample.get("instruction", "").strip()
        
        # 1. Exact duplicate check
        if instruction in seen_instructions:
            duplicates_removed += 1
            continue
            
        # 2. Near duplicate check (>90% overlap)
        words = get_words(instruction)
        is_duplicate = False
        
        # Only check against previous if words is not too small
        if len(words) >= 5:
            for prev_words in processed_word_sets:
                if len(prev_words) >= 5:
                    intersection = len(words & prev_words)
                    min_len = min(len(words), len(prev_words))
                    if min_len > 0 and (intersection / min_len) > 0.90:
                        is_duplicate = True
                        break
        
        if is_duplicate:
            duplicates_removed += 1
            continue
            
        # If we made it here, it's unique
        seen_instructions.add(instruction)
        processed_word_sets.append(words)
        unique_data.append(sample)
        unique_sources.append(source_tracking[i])

    source_counts["duplicates_removed"] = duplicates_removed

    # STEP 6: Shuffle and save
    print(f"Shuffling and saving...")
    
    # Zip together to keep source tracking aligned during shuffle
    combined = list(zip(unique_data, unique_sources))
    random.seed(42)
    random.shuffle(combined)
    
    final_data = [item[0] for item in combined]
    final_sources = [item[1] for item in combined]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in final_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    # Save source breakdown
    breakdown = []
    for sample, src in zip(final_data, final_sources):
        breakdown.append({
            "instruction": sample.get("instruction", ""),
            "source": src
        })
        
    with open(SOURCES_BREAKDOWN_FILE, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2, ensure_ascii=False)
        
    print(f"\n{'='*40}")
    print(f" DATASET MERGE SUMMARY")
    print(f"{'='*40}")
    print(f"Source 1 (generated clean): {source_counts['generated_clean']} samples")
    print(f"Source 2 (reasoning):       {source_counts['reasoning']} samples")
    print(f"Source 3 (transition):      {source_counts['transition']} samples")
    print(f"Source 4 (CSV converted):   {source_counts['csv_converted']} samples")
    print(f"Duplicates removed:         {source_counts['duplicates_removed']} samples")
    print(f"-----------------------------")
    print(f"FINAL TOTAL:                {len(final_data)} samples")
    print(f"{'='*40}")
    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Saved tracking to: {SOURCES_BREAKDOWN_FILE}\n")

if __name__ == "__main__":
    main()
