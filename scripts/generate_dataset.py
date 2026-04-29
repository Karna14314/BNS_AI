"""
generate_dataset.py - Indian Legal Training Data Generator (v2.0)

Generates high-quality BNS/BNSS/BSA training data for LLM fine-tuning.
Reads markdown source files, chunks them by legal sections, validates
content, sends to LLM for scenario generation, and validates outputs.

Usage:
    python generate_dataset.py              # Full pipeline run
    python generate_dataset.py --dry-run    # Process 5 chunks, preview only
    python generate_dataset.py --stats      # Show current dataset statistics
"""

import re
import os
import sys
import json
import time
import argparse
import random
from datetime import datetime
from collections import Counter

from openai import OpenAI
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; user can set env vars directly

# ============================================================
# Configuration
# ============================================================
INPUT_DIR = "extracted_md"
CHUNK_DIR = "data_chunks"
OUTPUT_FILE = "training_data.jsonl"
DETAILED_LOG_FILE = "detailed_log.jsonl"
REJECTED_SCENARIOS_FILE = "rejected_scenarios.jsonl"
REJECTED_CHUNKS_LOG = "rejected_chunks.log"

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
API_KEY = os.getenv("NVIDIA_API_KEY", "")

MIN_CHUNK_SIZE = 150
MAX_CHUNK_SIZE = 2500
RATE_LIMIT_SLEEP = 2
MAX_RETRIES = 5
BASE_WAIT_TIME = 10
DRY_RUN_LIMIT = 5

# Source markdown files (checked in both INPUT_DIR and current directory)
MD_FILE_CONFIGS = [
    {"filename": "BNS Book_After Correction.md", "prefix": "BNS"},
    {"filename": "Final_BNSS.md", "prefix": "BNSS"},
    {"filename": "Primar on New Criminal Laws.md", "prefix": "Primer"},
]

# ============================================================
# Chunk Validation
# ============================================================
def is_valid_chunk(text: str) -> tuple:
    """Validate that a chunk contains genuine BNS/BNSS/BSA legal content.

    Returns:
        (bool, str): (is_valid, reason)
    """
    if len(text.strip()) < MIN_CHUNK_SIZE:
        return False, "too_short"

    text_lower = text.lower()

    # Skip pure garbage chunks
    garbage_signals = [
        "picture intentionally omitted" in text_lower,
        text_lower.count("the bharatiya nyaya sanhita, 2023") > 5,
        len(re.findall(r'\b[ivxlc]+\b', text_lower)) > 20,
        "table of contents" in text_lower,
        "colour coding" in text_lower or "color coding" in text_lower,
    ]
    if sum(garbage_signals) >= 2:
        return False, "garbage_content"

    # Chunks that start with a numbered legal heading (## **N. TITLE**)
    # are inherently valid — they ARE sections of BNS/BNSS
    starts_with_legal_heading = bool(re.match(
        r'^(\[Continued from|#{1,4}\s+\*{0,2}(SECTION\s+\d+|CHAPTER\s*[-\s]*[IVXLC\d]+|\d+\.))',
        text.strip(), re.IGNORECASE
    ))
    if starts_with_legal_heading:
        return True, "valid"

    # For non-heading chunks, check for law references
    law_mentions = sum([
        "bharatiya nyaya sanhita" in text_lower,
        "bharatiya nagarik suraksha" in text_lower,
        "bharatiya sakshya" in text_lower,
        " bns " in text_lower or text_lower.startswith("bns"),
        " bnss " in text_lower or text_lower.startswith("bnss"),
        " bsa " in text_lower or text_lower.startswith("bsa"),
        "sanhita" in text_lower,
        "adhiniyam" in text_lower,
        bool(re.search(r'section \d+', text_lower)),
    ])

    # Also accept chunks with common criminal law terminology
    legal_terms = sum([
        "offence" in text_lower or "offense" in text_lower,
        "punishment" in text_lower,
        "imprisonment" in text_lower,
        "magistrate" in text_lower,
        "cognizable" in text_lower,
        "bail" in text_lower,
        "accused" in text_lower,
        "prosecution" in text_lower,
        "warrant" in text_lower,
        "fir" in text_lower or "first information report" in text_lower,
    ])

    if law_mentions >= 1:
        return True, "valid"
    if legal_terms >= 3:
        return True, "valid"

    return False, "no_law_reference"


# ============================================================
# Post-Generation Scenario Validation
# ============================================================
FORBIDDEN_REFS = [
    "Indian Penal Code", "IPC", " CrPC ", "Evidence Act",
    "Contract Act", "Companies Act", "Copyright Act",
    "Section 65B of BNSS",
    "Not determinable", "cannot be determined",
    "insufficient_content",
]

HANDBOOK_PHRASES = [
    "handbook",
    "colour coding",
    "color coding", 
    "green highlight",
    "blue highlight",
    "red highlight",
    "BPR&D",
    "Bureau of Police Research",
    "handbook's guidance",
    "training material",
    "as highlighted in",
    "colour-coded",
]


def validate_scenario(scenario: dict) -> tuple:
    """Validate a single LLM-generated scenario before saving.

    Returns:
        (bool, str): (is_valid, reason)
    """
    if "error" in scenario:
        return False, "model_flagged_bad_chunk"

    required_keys = {"chunk_source", "instruction", "input", "output"}
    if not required_keys.issubset(scenario.keys()):
        missing = required_keys - scenario.keys()
        return False, f"missing_keys: {missing}"

    combined_text = scenario.get("instruction", "") + scenario.get("output", "")

    for ref in FORBIDDEN_REFS:
        if ref.lower() in combined_text.lower():
            return False, f"forbidden_reference: {ref}"

    if len(scenario.get("instruction", "")) < 50:
        return False, "instruction_too_short"

    if len(scenario.get("output", "")) < 100:
        return False, "output_too_short"

    # Must mention at least one valid BNS/BNSS/BSA section reference
    has_valid_section = bool(re.search(
        r'Section\s+\d+[\(\d+\)]*\s*(of\s+)?(BNS|BNSS|BSA)'  
        r'|'
        r'(BNS|BNSS|BSA)\s+Section\s+\d+',
        combined_text, re.IGNORECASE
    ))
    if not has_valid_section:
        return False, "no_valid_section_reference"

    return True, "valid"


# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """You are a strict Indian criminal law expert generating \
training data for an AI legal assistant. You must follow ALL rules below \
with zero exceptions.

TASK: Generate exactly 3 legal scenarios based ONLY on the provided chunk.

OUTPUT FORMAT: Return ONLY a valid JSON array. No preamble, no explanation, \
no markdown. Start directly with [ and end with ].

EACH SCENARIO MUST HAVE EXACTLY THESE KEYS:
{
  "chunk_source": "<filename provided>",
  "scenario_type": "<one of: arrest_procedure|bail_application|evidence_admissibility|section_identification|punishment|trial_procedure|rights_of_accused>",
  "instruction": "<realistic question a lawyer, police officer, judge, or citizen would ask - 2-4 sentences describing the factual situation>",
  "input": "",
  "output": "<detailed answer with: 1) applicable section number and full name, 2) what the law says, 3) legal consequence or procedure, 4) any relevant Supreme Court position if mentioned in chunk>"
}

STRICT RULES - VIOLATING ANY RULE MAKES THE RESPONSE INVALID:
1. ONLY cite sections from BNS (Sections 1-358), BNSS (Sections 1-531), \
or BSA (Sections 1-170). NEVER cite IPC, CrPC, Evidence Act, \
Contract Act, Companies Act, or any other law.
2. NEVER invent section numbers. If the chunk does not explicitly mention \
a section number, write "Section not specified in source" - do not guess.
3. NEVER generate scenarios about handbook metadata, color coding systems, \
table of contents, preface, or document formatting.
4. Each scenario must describe a REAL situation: an arrest, a trial, a \
bail hearing, an FIR filing, evidence collection, etc.
5. The instruction must be a question that a real person would ask, not \
an abstract legal analysis request.
6. If the chunk contains insufficient BNS/BNSS/BSA content to generate \
3 valid scenarios, return: [{"error": "insufficient_content", \
"chunk_source": "<filename>"}]
7. Do NOT generate scenarios where the applicable section is \
"Not determinable" - skip those and generate different ones.

EXAMPLE OF GOOD OUTPUT:
[
  {
    "chunk_source": "BNSS_chunk_045.txt",
    "scenario_type": "arrest_procedure",
    "instruction": "A police officer arrests Ramesh for theft but does not inform him of the grounds of arrest, claiming urgency. Ramesh's family is not notified. Is this arrest lawful under BNSS?",
    "input": "",
    "output": "Under Section 47 of BNSS, every person arrested must be immediately informed of the grounds of arrest. This is a constitutional mandate under Article 22(1). Additionally, Section 50 of BNSS requires the police to inform a nominated person of the arrest and place of custody. Failure to comply with Section 47 renders the detention illegal and the arrested person is entitled to seek immediate release through a habeas corpus petition under Article 226. The Supreme Court in D.K. Basu v. State of West Bengal laid down guidelines that are now codified in BNSS."
  }
]

EXAMPLE OF BAD OUTPUT (NEVER DO THIS):
- Citing 'Section 65B of BNSS' (wrong - this is from old Evidence Act)
- Citing 'Indian Penal Code Section 420' (wrong law entirely)
- Scenario about 'colour coding system in the handbook' (not legal content)
- 'applicable_section: Not determinable from chunk' (useless for training)
"""


# ============================================================
# Phase 1: Chunking
# ============================================================

# Patterns that match ACTUAL headings in these markdown files
# BNSS uses: ## **1. SHORT TITLE...**, ## **42. PROTECTION OF...**
# BNS uses:  ## **SECTION 3. GENERAL EXPLANATIONS.**
# Both use:  ## **CHAPTER I (CHAPTER I, II)**
CHAPTER_PATTERN = re.compile(
    r'^#{1,4}\s+\*{0,2}CHAPTER\s*[-\s]*[IVXLC\d]+',
    re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r'^#{1,4}\s+\*{0,2}SECTION\s+\d+[\.\s]',
    re.IGNORECASE
)

# Numbered section headings like: ## **42. PROTECTION OF MEMBERS...**
NUMBERED_SECTION_PATTERN = re.compile(
    r'^#{1,4}\s+\*{0,2}\d+\.\s+[A-Z]'
)

# Garbage lines to skip anywhere in the document
GARBAGE_PATTERNS = [
    re.compile(r'picture \[\d+ x \d+\] intentionally omitted', re.IGNORECASE),
    re.compile(r'^[ivxlc]+\s*$', re.IGNORECASE),       # Roman numerals alone
    re.compile(r'iqfyl|vuqla|Hkkjr|lafo/kku'),          # Hindi OCR artifacts
    re.compile(r'^\d+\s*$'),                             # Bare page numbers
    re.compile(r'^-{3,}\s*$'),                           # Horizontal rules
    re.compile(r'Start of picture text|End of picture text', re.IGNORECASE),
]

# Primer-specific topic headings (## **Definition**, ## **Punishment**, etc.)
# These don't match CHAPTER/SECTION/numbered patterns but are valid split points
PRIMER_TOPIC_PATTERN = re.compile(
    r'^#{1,4}\s+\*{0,2}(Definition|Punishment|Scheme|Offence|'
    r'Arrest|Bail|Evidence|Trial|Appeal|Sentence|Prosecution|'
    r'Search|Seizure|Remand|Summons|Warrant|Complaint|'
    r'PART\s+[IVXLC\d]+|'
    r'BHARATIYA\s+(NYAYA|NAGARIK|SAKSHYA)|'
    r'NEW OFFENCES|REPEALED|Major Changes|COMPARATIVE CHART|'
    r'NEW CRIMINAL LAWS|INTRODUCTION)',
    re.IGNORECASE
)

# Headings that should NOT be treated as split points (stay in current chunk)
SKIP_HEADINGS = [
    'preface', 'table of contents', 'executive summary',
    'colour coding', 'color coding', 'background',
    'highlights of', 'comparison with', 'illustration',
    'sections merged', 'bureau of police', 'foreword',
    'acknowledgment', 'the colour coding',
    'modification', 'modifcation',  # typos in source
    'addition of subsection', 'addition of section',
    '## **on**', '## **primer**',  # Primer title fragments
]


def resolve_md_files() -> list:
    """Find markdown source files in INPUT_DIR or current directory."""
    found = []
    for cfg in MD_FILE_CONFIGS:
        path_in_dir = os.path.join(INPUT_DIR, cfg["filename"])
        if os.path.exists(path_in_dir):
            found.append({"path": path_in_dir, "prefix": cfg["prefix"]})
        elif os.path.exists(cfg["filename"]):
            found.append({"path": cfg["filename"], "prefix": cfg["prefix"]})
        else:
            print(f"  [WARN] Source file not found: {cfg['filename']}")
    return found


def _is_garbage_line(line: str) -> bool:
    """Check if a line is garbage (pictures, page numbers, OCR artifacts)."""
    for pattern in GARBAGE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def _is_skip_heading(line: str) -> bool:
    """Check if a heading should NOT be a split point."""
    line_lower = line.lower().strip()
    return any(skip in line_lower for skip in SKIP_HEADINGS)


def _is_split_point(line: str) -> bool:
    """Check if this line is a legal section/chapter heading that should start a new chunk."""
    stripped = line.strip()
    if not stripped.startswith('#'):
        return False
    if _is_skip_heading(stripped):
        return False
    return (bool(CHAPTER_PATTERN.match(stripped)) or
            bool(SECTION_PATTERN.match(stripped)) or
            bool(NUMBERED_SECTION_PATTERN.match(stripped)) or
            bool(PRIMER_TOPIC_PATTERN.match(stripped)))


def _save_chunk(prefix: str, counter: int, lines: list) -> str:
    """Save a chunk to disk. Returns the filename."""
    chunk_filename = f"{prefix}_chunk_{counter:04d}.txt"
    chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
    if not os.path.exists(chunk_path):
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
    return chunk_filename


def chunk_markdown_files(md_files: list) -> dict:
    """Split markdown files into validated legal chunks.

    Strategy:
    1. Skip preamble (everything before first CHAPTER/SECTION heading)
    2. Split on legal headings (CHAPTER, SECTION, numbered sections)
    3. Sub-split oversized chunks at subsection markers **(N)**
    4. Filter garbage lines throughout
    5. Validate each chunk with is_valid_chunk()

    Returns:
        dict with counts: total_raw, valid, rejected, rejection_reasons
    """
    import shutil
    if os.path.exists(CHUNK_DIR):
        shutil.rmtree(CHUNK_DIR)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    stats = {"total_raw": 0, "valid": 0, "rejected": 0, "rejection_reasons": Counter()}
    rejected_log_entries = []

    for md_info in md_files:
        md_path = md_info["path"]
        prefix = md_info["prefix"]
        print(f"  Processing: {os.path.basename(md_path)} (prefix={prefix})")

        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        current_chunk_lines = []
        current_heading = ""
        chunk_counter = 0
        in_preamble = True

        for line in lines:
            # Detect end of preamble — first real CHAPTER or SECTION heading
            if in_preamble:
                if _is_split_point(line):
                    in_preamble = False
                    # Fall through to start first chunk
                else:
                    continue

            # Skip garbage lines anywhere in document
            if _is_garbage_line(line):
                continue

            # Check if this line starts a new chunk
            if _is_split_point(line):
                # Save the previous chunk if it has content
                if current_chunk_lines:
                    chunk_text = "".join(current_chunk_lines).strip()
                    stats["total_raw"] += 1
                    is_ok, reason = is_valid_chunk(chunk_text)
                    if is_ok:
                        chunk_counter += 1
                        _save_chunk(prefix, chunk_counter, current_chunk_lines)
                        stats["valid"] += 1
                    else:
                        stats["rejected"] += 1
                        stats["rejection_reasons"][reason] += 1
                        preview = chunk_text[:120].replace("\n", " ")
                        rejected_log_entries.append(
                            f"[{prefix}] reason={reason} | preview: {preview}..."
                        )

                # Start new chunk with this heading
                current_chunk_lines = [line]
                current_heading = line.strip()
            else:
                current_chunk_lines.append(line)

                # Sub-split if chunk gets too large
                current_text = "".join(current_chunk_lines)
                if len(current_text) > MAX_CHUNK_SIZE:
                    # Look for a natural break — subsection marker **(N)**
                    search_start = len(current_text) // 2
                    subsection_break = re.search(
                        r'\n\*\*\(\d+\)\*\*',
                        current_text[search_start:]
                    )
                    if subsection_break:
                        split_char_pos = search_start + subsection_break.start()
                        # Split the lines at this position
                        accumulated = 0
                        split_line_idx = 0
                        for idx, ln in enumerate(current_chunk_lines):
                            accumulated += len(ln)
                            if accumulated >= split_char_pos:
                                split_line_idx = idx
                                break

                        if split_line_idx > 0:
                            part1 = current_chunk_lines[:split_line_idx]
                            part1_text = "".join(part1).strip()
                            stats["total_raw"] += 1
                            is_ok, reason = is_valid_chunk(part1_text)
                            if is_ok:
                                chunk_counter += 1
                                _save_chunk(prefix, chunk_counter, part1)
                                stats["valid"] += 1
                            else:
                                stats["rejected"] += 1
                                stats["rejection_reasons"][reason] += 1

                            # Continue with remainder
                            current_chunk_lines = [
                                f"[Continued from {current_heading}]\n"
                            ] + current_chunk_lines[split_line_idx:]

        # Save the last chunk
        if current_chunk_lines:
            chunk_text = "".join(current_chunk_lines).strip()
            stats["total_raw"] += 1
            is_ok, reason = is_valid_chunk(chunk_text)
            if is_ok:
                chunk_counter += 1
                _save_chunk(prefix, chunk_counter, current_chunk_lines)
                stats["valid"] += 1
            else:
                stats["rejected"] += 1
                stats["rejection_reasons"][reason] += 1

        print(f"    -> {chunk_counter} valid chunks from {os.path.basename(md_path)}")

    # Write rejection log
    with open(REJECTED_CHUNKS_LOG, "w", encoding="utf-8") as f:
        f.write(f"=== Rejected Chunks Log - {datetime.now().isoformat()} ===\n\n")
        f.write("\n".join(rejected_log_entries))

    return stats


# ============================================================
# Phase 2: LLM Processing
# ============================================================
def get_processed_chunks(output_file: str) -> set:
    """Get set of chunk filenames already processed (for resumability)."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "chunk_source" in data:
                    processed.add(data["chunk_source"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def clean_json_response(response_text: str) -> str:
    """Strip markdown code fences and whitespace from LLM response."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def process_chunks_with_llm(dry_run: bool = False) -> dict:
    """Send validated chunks to LLM and validate generated scenarios.

    Args:
        dry_run: If True, only process DRY_RUN_LIMIT chunks and print output.

    Returns:
        dict with processing statistics.
    """
    if not API_KEY:
        print("\n[ERROR] NVIDIA_API_KEY not found!")
        print("  Set it in a .env file:  NVIDIA_API_KEY=nvapi-xxxxx")
        print("  Or set the environment variable directly.")
        sys.exit(1)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    if not os.path.exists(CHUNK_DIR):
        print(f"\n[ERROR] Chunk directory '{CHUNK_DIR}' not found. Run Phase 1 first.")
        return {}

    all_chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".txt")])

    # For resumability, check which chunks are already in the detailed log
    processed_chunks = get_processed_chunks(DETAILED_LOG_FILE)
    chunks_to_process = [c for c in all_chunks if c not in processed_chunks]

    if dry_run:
        chunks_to_process = chunks_to_process[:DRY_RUN_LIMIT]
        print(f"\n{'='*60}")
        print(f"  DRY RUN MODE — Processing {len(chunks_to_process)} chunks")
        print(f"{'='*60}")

    print(f"\n  Status Report:")
    print(f"  - Total valid chunks:    {len(all_chunks)}")
    print(f"  - Already processed:     {len(processed_chunks)}")
    print(f"  - Remaining to process:  {len(chunks_to_process)}")

    if not chunks_to_process:
        print("\n  All chunks have been processed! Nothing to do.")
        return {"processed": 0, "valid": 0, "rejected": 0, "reasons": Counter()}

    stats = {"processed": 0, "valid": 0, "rejected": 0, "reasons": Counter()}
    pbar = tqdm(chunks_to_process, desc="Generating Scenarios", unit="chunk")

    for chunk_filename in pbar:
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()

        user_message = (
            f"Chunk Filename: {chunk_filename}\n\n"
            f"Chunk Text:\n{chunk_text}"
        )

        # Retry with exponential backoff
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=2048,
                    stream=False,
                )

                output_text = completion.choices[0].message.content
                cleaned_output = clean_json_response(output_text)

                try:
                    scenarios = json.loads(cleaned_output)
                    if not isinstance(scenarios, list):
                        scenarios = [scenarios]
                except json.JSONDecodeError:
                    pbar.set_postfix_str(f"JSON parse error on {chunk_filename}")
                    break

                # Validate and route each scenario
                for scenario in scenarios:
                    scenario["chunk_source"] = chunk_filename
                    is_ok, reason = validate_scenario(scenario)

                    # Detailed log always gets everything
                    log_entry = {
                        **scenario,
                        "validation_status": "valid" if is_ok else "rejected",
                        "validation_reason": reason,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if not dry_run:
                        with open(DETAILED_LOG_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                    if is_ok:
                        # Alpaca format for training
                        training_entry = {
                            "instruction": scenario["instruction"],
                            "input": scenario.get("input", ""),
                            "output": scenario["output"],
                        }
                        if dry_run:
                            print(f"\n  [VALID] {chunk_filename}")
                            print(f"    Q: {scenario['instruction'][:100]}...")
                            print(f"    A: {scenario['output'][:100]}...")
                        else:
                            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")
                        stats["valid"] += 1
                    else:
                        if dry_run:
                            print(f"\n  [REJECTED] {chunk_filename} — {reason}")
                        else:
                            with open(REJECTED_SCENARIOS_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        stats["rejected"] += 1
                        stats["reasons"][reason] += 1

                stats["processed"] += 1
                success = True
                break

            except Exception as e:
                wait_time = BASE_WAIT_TIME * (2 ** attempt)
                pbar.set_postfix_str(f"API error, retry in {wait_time}s...")
                time.sleep(wait_time)

        if success:
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            pbar.set_postfix_str(f"FAILED: {chunk_filename}")
            stats["reasons"]["api_failure"] += 1

    return stats


# ============================================================
# Phase 3: Reporting
# ============================================================
def generate_report(chunk_stats: dict = None, llm_stats: dict = None):
    """Print a summary report of the pipeline run."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE REPORT")
    print(f"{'='*60}")

    if chunk_stats:
        print(f"\n  Phase 1 — Chunking:")
        print(f"    Raw chunks extracted:    {chunk_stats['total_raw']}")
        print(f"    Valid chunks saved:       {chunk_stats['valid']}")
        print(f"    Rejected chunks:          {chunk_stats['rejected']}")
        if chunk_stats["rejection_reasons"]:
            print(f"    Rejection breakdown:")
            for reason, count in chunk_stats["rejection_reasons"].most_common():
                print(f"      {reason}: {count}")

    if llm_stats and llm_stats.get("processed", 0) > 0:
        print(f"\n  Phase 2 — LLM Generation:")
        print(f"    Chunks processed:         {llm_stats['processed']}")
        print(f"    Valid scenarios:           {llm_stats['valid']}")
        print(f"    Rejected scenarios:        {llm_stats['rejected']}")
        if llm_stats["reasons"]:
            print(f"    Rejection breakdown:")
            for reason, count in llm_stats["reasons"].most_common():
                print(f"      {reason}: {count}")

        total_valid = llm_stats["valid"]
        if total_valid >= 500:
            print(f"\n  [OK] Fine-tuning readiness: GOOD ({total_valid} samples)")
        elif total_valid >= 200:
            print(f"\n  [!!] Fine-tuning readiness: MARGINAL ({total_valid}/500 target)")
        else:
            print(f"\n  [XX] Fine-tuning readiness: INSUFFICIENT ({total_valid}/500 target)")

    print(f"\n{'='*60}\n")


def show_stats():
    """Show current dataset statistics from existing output files."""
    print(f"\n{'='*60}")
    print(f"  CURRENT DATASET STATISTICS")
    print(f"{'='*60}")

    # Count chunks
    if os.path.exists(CHUNK_DIR):
        chunks = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".txt")]
        print(f"\n  Valid chunks in '{CHUNK_DIR}': {len(chunks)}")
        prefixes = Counter(f.split("_chunk_")[0] for f in chunks if "_chunk_" in f)
        for prefix, count in prefixes.most_common():
            print(f"    {prefix}: {count}")
    else:
        print(f"\n  No chunk directory found.")

    # Count training samples
    for filepath, label in [
        (OUTPUT_FILE, "Training samples"),
        (DETAILED_LOG_FILE, "Detailed log entries"),
        (REJECTED_SCENARIOS_FILE, "Rejected scenarios"),
    ]:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            print(f"\n  {label} ({filepath}): {count}")
        else:
            print(f"\n  {label} ({filepath}): 0 (file not created yet)")

    # Rejection breakdown from rejected scenarios
    if os.path.exists(REJECTED_SCENARIOS_FILE):
        reasons = Counter()
        with open(REJECTED_SCENARIOS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    reasons[data.get("validation_reason", "unknown")] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if reasons:
            print(f"\n  Rejection reasons breakdown:")
            for reason, count in reasons.most_common():
                print(f"    {reason}: {count}")

    # Fine-tuning readiness
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
        if total >= 500:
            print(f"\n  [OK] Fine-tuning readiness: GOOD ({total} samples)")
        elif total >= 200:
            print(f"\n  [!!] Fine-tuning readiness: MARGINAL ({total}/500 target)")
        else:
            print(f"\n  [XX] Fine-tuning readiness: INSUFFICIENT ({total}/500 target)")

    print(f"\n{'='*60}\n")


def clean_dataset():
    """Post-processing filter to clean handbook-meta samples from training_data.jsonl"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"\n  [ERROR] Source file {OUTPUT_FILE} not found!")
        return

    clean_file = "training_data_clean.jsonl"
    removed_count = 0
    kept_count = 0

    with open(OUTPUT_FILE, "r", encoding="utf-8") as infile, \
         open(clean_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                scenario = json.loads(line)
                combined_text = scenario.get("instruction", "") + " " + scenario.get("output", "")
                
                has_handbook_phrase = False
                for phrase in HANDBOOK_PHRASES:
                    if phrase.lower() in combined_text.lower():
                        has_handbook_phrase = True
                        break
                
                if has_handbook_phrase:
                    removed_count += 1
                else:
                    outfile.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                    kept_count += 1
                    
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*60}")
    print(f"  DATASET CLEANUP REPORT")
    print(f"{'='*60}")
    print(f"  Removed {removed_count} handbook-meta samples.")
    print(f"  Clean dataset: {kept_count} samples remaining.")
    print(f"  Saved to: {clean_file}")
    print(f"{'='*60}\n")


def run_test_10():
    """Diagnostic test: Select 10 diverse chunks and run through LLM without saving."""
    if not API_KEY:
        print("\n[ERROR] NVIDIA_API_KEY not found!")
        return

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    if not os.path.exists(CHUNK_DIR):
        print(f"\n[ERROR] Chunk directory '{CHUNK_DIR}' not found. Run Phase 1 first.")
        return

    all_chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".txt")])
    bns_chunks = [c for c in all_chunks if c.startswith("BNS_")]
    bnss_chunks = [c for c in all_chunks if c.startswith("BNSS_")]
    primer_chunks = [c for c in all_chunks if c.startswith("Primer_")]

    random.seed(42)
    selected = []

    # Select 3 BNS
    if len(bns_chunks) >= 3:
        selected.extend(random.sample(bns_chunks, 3))
    else:
        selected.extend(bns_chunks)

    # Select 3 BNSS
    if len(bnss_chunks) >= 3:
        selected.extend(random.sample(bnss_chunks, 3))
    else:
        selected.extend(bnss_chunks)

    # Select 2 Primer
    if len(primer_chunks) >= 2:
        selected.extend(random.sample(primer_chunks, 2))
    else:
        selected.extend(primer_chunks)

    # Remaining to reach 10
    remaining_pool = [c for c in all_chunks if c not in selected]
    needed = 10 - len(selected)
    if len(remaining_pool) >= needed:
        selected.extend(random.sample(remaining_pool, needed))
    else:
        selected.extend(remaining_pool)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC TEST — Processing {len(selected)} diverse chunks")
    print(f"  (No files will be updated)")
    print(f"{'='*60}\n")

    test_stats = {"passed": 0, "failed": 0, "reasons": Counter()}

    for i, chunk_filename in enumerate(selected, 1):
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()

        print(f"========================================")
        print(f"SAMPLE {i}/{len(selected)} | Source: {chunk_filename}")
        print(f"========================================")

        user_message = f"Chunk Filename: {chunk_filename}\n\nChunk Text:\n{chunk_text}"

        success = False
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                top_p=0.7,
                max_tokens=2048,
            )

            output_text = completion.choices[0].message.content
            cleaned_output = clean_json_response(output_text)
            scenarios = json.loads(cleaned_output)
            if not isinstance(scenarios, list):
                scenarios = [scenarios]

            for scenario in scenarios:
                is_ok, reason = validate_scenario(scenario)
                print(f"SCENARIO TYPE : {scenario.get('scenario_type', 'N/A')}")
                print(f"\nINSTRUCTION   : {scenario.get('instruction', 'N/A')}")
                print(f"\nOUTPUT        : {scenario.get('output', 'N/A')}")
                print(f"\nVALIDATION    : {'PASS' if is_ok else 'FAIL'} ({reason if not is_ok else 'valid'})")
                print(f"{'-'*40}")

                if is_ok:
                    test_stats["passed"] += 1
                else:
                    test_stats["failed"] += 1
                    test_stats["reasons"][reason] += 1

            success = True
        except Exception as e:
            print(f"API or Parse Error: {e}")

        if i < len(selected):
            time.sleep(RATE_LIMIT_SLEEP)

    # Print Summary
    print(f"\n=================== TEST SUMMARY ===================")
    total = test_stats["passed"] + test_stats["failed"]
    print(f"Passed  : {test_stats['passed']}/{total}")
    print(f"Failed  : {test_stats['failed']}/{total}")

    if test_stats["reasons"]:
        print(f"\nFailure reasons:")
        for reason, count in test_stats["reasons"].most_common():
            print(f"- {reason}: {count}")

    # Recommendation
    pass_rate = test_stats["passed"] / total if total > 0 else 0
    print(f"\nRecommendation:")
    if pass_rate >= 0.8:
        print("- Ready for full run. Execute: python generate_dataset.py")
    elif pass_rate >= 0.6:
        print("- Adjust system prompt before full run. Review failures above.")
    else:
        print("- STOP. Major issue with chunking or prompt. Fix before proceeding.")
    print(f"=====================================================\n")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Indian legal training data for BNS/BNSS/BSA fine-tuning."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=f"Process only {DRY_RUN_LIMIT} chunks and preview output without saving."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show current dataset statistics and exit."
    )
    parser.add_argument(
        "--count-chunks", action="store_true",
        help="Run only Phase 1 (chunking) and print chunk counts. No LLM calls."
    )
    parser.add_argument(
        "--test-10", action="store_true",
        help="Diagnostic: select 10 diverse chunks and run through LLM to verify quality."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Post-processing: clean handbook-meta samples from training_data.jsonl."
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.clean:
        clean_dataset()
        return

    print(f"\n{'='*60}")
    print(f"  BNS/BNSS/BSA Training Data Generator v2.0")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Verify API key (not needed for --count-chunks)
    if not API_KEY and not args.count_chunks:
        print("\n  [WARN] NVIDIA_API_KEY not found in environment or .env file.")
        print("  Create a .env file with: NVIDIA_API_KEY=nvapi-xxxxx")
        print("  Or set the NVIDIA_API_KEY environment variable.")

    # --- Phase 1: Chunking ---
    print(f"\n  Phase 1: Chunking markdown files...")
    md_files = resolve_md_files()
    if not md_files:
        print("  [ERROR] No source markdown files found!")
        print(f"  Looked in: '{INPUT_DIR}/' and current directory.")
        sys.exit(1)

    chunk_stats = chunk_markdown_files(md_files)
    print(f"\n  Phase 1 Complete: {chunk_stats['valid']} valid chunks, "
          f"{chunk_stats['rejected']} rejected.")

    # If --test-10, run diagnostic and exit
    if args.test_10:
        run_test_10()
        return

    # If --count-chunks, print detailed breakdown and exit
    if args.count_chunks:
        print(f"\n  --- Chunk Analysis ---")
        if os.path.exists(CHUNK_DIR):
            all_chunks = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".txt")]
            bns = [c for c in all_chunks if c.startswith("BNS_")]
            bnss = [c for c in all_chunks if c.startswith("BNSS_")]
            primer = [c for c in all_chunks if c.startswith("Primer_")]
            print(f"  BNS chunks:    {len(bns)}")
            print(f"  BNSS chunks:   {len(bnss)}")
            print(f"  Primer chunks: {len(primer)}")
            print(f"  Total:         {len(all_chunks)}")
        if chunk_stats["rejection_reasons"]:
            print(f"\n  Rejection breakdown:")
            for reason, count in chunk_stats["rejection_reasons"].most_common():
                print(f"    {reason}: {count}")
        print(f"\n  See {REJECTED_CHUNKS_LOG} for details.")
        return

    # --- Phase 2: LLM Generation ---
    print(f"\n  Phase 2: Generating scenarios with LLM...")
    llm_stats = process_chunks_with_llm(dry_run=args.dry_run)

    # --- Phase 3: Report ---
    generate_report(chunk_stats, llm_stats)

    if args.dry_run:
        print("  (Dry run complete -- no files were modified.)")
    else:
        valid = llm_stats.get("valid", 0)
        rejected = llm_stats.get("rejected", 0)
        print(f"  Generated {valid} valid samples. Rejected {rejected} samples.")
        if llm_stats.get("reasons"):
            print(f"  Rejection breakdown: {dict(llm_stats['reasons'])}")

    print(f"\n  Pipeline execution complete!")


if __name__ == "__main__":
    main()
