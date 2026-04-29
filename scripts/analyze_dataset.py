import json
import random
import re
import os
from collections import defaultdict

# Configuration
REJECTED_FILE = "rejected_scenarios.jsonl"
ACCEPTED_FILE = "training_data.jsonl"

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

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def analyze_rejected():
    print_separator("1. REJECTED SCENARIOS ANALYSIS")
    if not os.path.exists(REJECTED_FILE):
        print(f"File {REJECTED_FILE} not found.")
        return
    
    rejected = []
    with open(REJECTED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rejected.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    print(f"Total rejected count: {len(rejected)}")
    if not rejected:
        return
        
    reasons = defaultdict(list)
    for r in rejected:
        reason = r.get("validation_reason", "unknown")
        reasons[reason].append(r)
        
    print("\nBreakdown of rejection reasons:")
    for reason, items in sorted(reasons.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - {reason}: {len(items)}")
        
    for reason, items in sorted(reasons.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n--- Examples for reason: {reason} ---")
        sample_size = min(3, len(items))
        samples = random.sample(items, sample_size)
        for i, s in enumerate(samples, 1):
            print(f"\nExample {i}:")
            print(f"Instruction: {s.get('instruction', 'N/A')}")
            print(f"Output: {s.get('output', 'N/A')}")
            print(f"Reason: {s.get('validation_reason', 'N/A')}")

def check_suspicious(scenario):
    flags = []
    output = scenario.get("output", "")
    instruction = scenario.get("instruction", "")
    combined = (instruction + " " + output).lower()
    
    words = len(re.findall(r'\b\w+\b', output))
    if words < 80:
        flags.append(f"Short output ({words} words)")
        
    for phrase in ["handbook", "colour coding", "bpr&d", "color coding"]:
        if phrase in combined:
            flags.append(f"Contains '{phrase}'")
            
    for phrase in ["not specified in source", "not mentioned in chunk"]:
        if phrase in combined:
            flags.append(f"Contains '{phrase}'")
            
    for phrase in ["indian penal code", "ipc"]:
        if re.search(r'\bindian penal code\b|\bipc\b', combined):
            flags.append(f"Contains '{phrase}'")
            
    if not re.search(r'\d+', output):
        flags.append("No number pattern in output")
        
    return flags

def analyze_accepted():
    print_separator("2. ACCEPTED SCENARIOS ANALYSIS")
    if not os.path.exists(ACCEPTED_FILE):
        print(f"File {ACCEPTED_FILE} not found.")
        return [], []
        
    accepted = []
    with open(ACCEPTED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    accepted.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                    
    print(f"Total accepted count: {len(accepted)}")
    if not accepted:
        return [], []
        
    print("\n--- Random Sample of 20 Accepted Scenarios ---")
    sample_size = min(20, len(accepted))
    samples = random.sample(accepted, sample_size)
    for i, s in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Instruction: {s.get('instruction', 'N/A')}")
        
    flagged_samples = []
    for s in accepted:
        flags = check_suspicious(s)
        if flags:
            flagged_samples.append((s, flags))
            
    print(f"\n--- Suspicious Patterns ---")
    print(f"Flagged samples count: {len(flagged_samples)}")
    
    if flagged_samples:
        print("\nExamples of flagged samples:")
        sample_size = min(5, len(flagged_samples))
        examples = random.sample(flagged_samples, sample_size)
        for i, (s, flags) in enumerate(examples, 1):
            print(f"\nFlag Example {i}:")
            print(f"Flags: {', '.join(flags)}")
            print(f"Instruction: {s.get('instruction', 'N/A')}")
            print(f"Output: {s.get('output', 'N/A')}")
            
    return accepted, flagged_samples

def check_overlap(accepted):
    print_separator("3. OVERLAP CHECK (Near-Duplicate Instructions)")
    
    if len(accepted) < 2:
        print("Not enough accepted samples to check overlap.")
        return
        
    # Pre-compute word sets
    instructions = []
    for s in accepted:
        inst = s.get('instruction', '')
        words = get_words(inst)
        instructions.append((inst, words))
        
    duplicates = []
    n = len(instructions)
    for i in range(n):
        inst_i, words_i = instructions[i]
        if len(words_i) < 3: continue
        for j in range(i + 1, n):
            inst_j, words_j = instructions[j]
            if len(words_j) < 3: continue
            
            intersection = len(words_i & words_j)
            min_len = min(len(words_i), len(words_j))
            if min_len == 0: continue
            
            ratio = intersection / min_len
            if ratio > 0.7:
                duplicates.append((inst_i, inst_j, ratio))
                
    print(f"Near-duplicate pairs found: {len(duplicates)}")
    if duplicates:
        print("\nExamples of duplicate pairs:")
        duplicates.sort(key=lambda x: x[2], reverse=True)
        sample_size = min(5, len(duplicates))
        for i in range(sample_size):
            print(f"\nPair {i+1} (Overlap: {duplicates[i][2]:.1%}):")
            print(f"1: {duplicates[i][0]}")
            print(f"2: {duplicates[i][1]}")

def main():
    random.seed(42)
    analyze_rejected()
    accepted, _ = analyze_accepted()
    check_overlap(accepted)

if __name__ == '__main__':
    main()
