import json
import os
import re

# Configuration
INPUT_FILE = "training_data.jsonl"
OUTPUT_FILE = "training_data_final.jsonl"

def clean_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    original_count = 0
    removed_count = 0
    final_count = 0

    # Removal patterns
    removal_phrases = [
        "not specified in source",
        "cannot be determined",
        "section not specified",
        "handbook",
        "colour coding",
        "color coding"
    ]

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            original_count += 1
            try:
                scenario = json.loads(line)
                instruction = scenario.get("instruction", "")
                output = scenario.get("output", "")
                combined_lower = (instruction + " " + output).lower()
                
                # Rule 1-3 & 5: Phrases
                should_remove = False
                for phrase in removal_phrases:
                    if phrase in combined_lower:
                        should_remove = True
                        break
                
                if should_remove:
                    removed_count += 1
                    continue

                # Rule 4: Output length under 30 words
                words = re.findall(r'\b\w+\b', output)
                if len(words) < 30:
                    removed_count += 1
                    continue
                
                # Rule 6: Instruction contains "chunk_0"
                if "chunk_0" in instruction.lower():
                    removed_count += 1
                    continue

                # Keep everything else
                outfile.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                final_count += 1
                
            except json.JSONDecodeError:
                removed_count += 1
                continue

    print(f"\n{'='*40}")
    print(f" FINAL DATASET CLEANUP")
    print(f"{'='*40}")
    print(f"Original count : {original_count}")
    print(f"Removed count  : {removed_count}")
    print(f"Final count    : {final_count}")
    print(f"Saved to       : {OUTPUT_FILE}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    clean_dataset()
