#!/usr/bin/env python3
"""
PDF to Markdown Extractor & Chunker for LLM Training Dataset

This script converts PDFs to clean Markdown using pymupdf4llm and then
splits them into header-based chunks for LLM context windows.

Usage:
    python pdf_to_markdown_extractor.py
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure pymupdf4llm is installed
try:
    import pymupdf4llm
except ImportError:
    print("Error: pymupdf4llm not installed. Run: pip install pymupdf4llm")
    sys.exit(1)


# Configuration
PDF_FILES = [
    "BNS Book_After Correction.pdf",
    "Final_BNSS.pdf",
    "Primar on New Criminal Laws.pdf"
]

OUTPUT_MD_DIR = Path("extracted_md")
OUTPUT_CHUNKS_DIR = Path("chunks")

# Regex pattern to match markdown headers (## or ###)
HEADER_PATTERN = re.compile(r'^(##|###)\s+(.+)$', re.MULTILINE)


def ensure_directories():
    """Create output directories if they don't exist."""
    OUTPUT_MD_DIR.mkdir(exist_ok=True)
    OUTPUT_CHUNKS_DIR.mkdir(exist_ok=True)
    print(f"[INFO] Output directories ready: {OUTPUT_MD_DIR}, {OUTPUT_CHUNKS_DIR}")


def get_base_name(pdf_path: str) -> str:
    """Extract base name from PDF filename (without extension)."""
    return Path(pdf_path).stem.replace(" ", "_").replace(".", "_")


def convert_pdf_to_markdown(pdf_path: str) -> Tuple[bool, str, str]:
    """
    Convert a single PDF to Markdown using pymupdf4llm.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (success: bool, content: str, error_message: str)
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        return False, "", f"File not found: {pdf_path}"
    
    base_name = get_base_name(pdf_path)
    output_file = OUTPUT_MD_DIR / f"{base_name}.md"
    
    print(f"\n[INFO] Processing: {pdf_path}")
    print(f"[INFO] Output will be saved to: {output_file}")
    
    try:
        # Use pymupdf4llm to extract content with table preservation
        # This library handles headers, tables, and structure better than alternatives
        md_content = pymupdf4llm.to_markdown(str(pdf_file))
        
        if not md_content or not md_content.strip():
            return False, "", "No content extracted from PDF"
        
        # Write the markdown content to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        content_length = len(md_content)
        print(f"[SUCCESS] Extracted {content_length:,} characters to {output_file}")
        
        return True, str(output_file), ""
        
    except Exception as e:
        error_msg = f"Extraction failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, "", error_msg


def split_markdown_by_headers(md_file_path: str) -> Tuple[bool, int, str]:
    """
    Split a markdown file into chunks based on headers (## or ###).
    
    Args:
        md_file_path: Path to the markdown file
        
    Returns:
        Tuple of (success: bool, chunks_created: int, error_message: str)
    """
    md_path = Path(md_file_path)
    base_name = md_path.stem
    
    print(f"\n[INFO] Chunking: {md_path.name}")
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            return False, 0, "Markdown file is empty"
        
        # Find all header positions
        matches = list(HEADER_PATTERN.finditer(content))
        
        if not matches:
            # No headers found, treat entire content as one chunk
            chunks = [content]
            print(f"[WARN] No headers (## or ###) found. Treating entire file as single chunk.")
        else:
            # Split content at header positions
            chunks = []
            
            for i, match in enumerate(matches):
                start_pos = match.start()
                
                # Determine end position (start of next header or end of content)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(content)
                
                chunk = content[start_pos:end_pos].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
        
        # Save chunks
        chunks_created = 0
        for idx, chunk in enumerate(chunks, start=1):
            chunk_filename = f"{base_name}_chunk_{idx:03d}.txt"
            chunk_path = OUTPUT_CHUNKS_DIR / chunk_filename
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            chunks_created += 1
        
        print(f"[SUCCESS] Created {chunks_created} chunks from {md_path.name}")
        return True, chunks_created, ""
        
    except Exception as e:
        error_msg = f"Chunking failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return False, 0, error_msg


def process_all_pdfs():
    """
    Main processing pipeline: convert PDFs to Markdown, then chunk them.
    """
    print("=" * 70)
    print("PDF TO MARKDOWN EXTRACTOR & CHUNKER")
    print("For India's New Criminal Laws LLM Training Dataset")
    print("=" * 70)
    
    ensure_directories()
    
    # Track statistics
    stats = {
        "pdfs_processed": 0,
        "pdfs_failed": 0,
        "md_files_created": 0,
        "total_chunks": 0
    }
    
    successfully_converted = []
    
    # Phase 1: Convert PDFs to Markdown
    print("\n" + "-" * 70)
    print("PHASE 1: PDF TO MARKDOWN CONVERSION")
    print("-" * 70)
    
    for pdf_file in PDF_FILES:
        success, output_path, error = convert_pdf_to_markdown(pdf_file)
        
        if success:
            stats["pdfs_processed"] += 1
            stats["md_files_created"] += 1
            successfully_converted.append(output_path)
        else:
            stats["pdfs_failed"] += 1
            print(f"[FAILED] {pdf_file}: {error}")
    
    # Phase 2: Chunk the Markdown files
    print("\n" + "-" * 70)
    print("PHASE 2: MARKDOWN TO CHUNKS SPLITTING")
    print("-" * 70)
    
    for md_file in successfully_converted:
        success, chunks_count, error = split_markdown_by_headers(md_file)
        
        if success:
            stats["total_chunks"] += chunks_count
        else:
            print(f"[FAILED] Chunking {md_file}: {error}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  PDFs processed successfully: {stats['pdfs_processed']}")
    print(f"  PDFs failed: {stats['pdfs_failed']}")
    print(f"  Markdown files created: {stats['md_files_created']}")
    print(f"  Total chunks created: {stats['total_chunks']}")
    print(f"  Output directories:")
    print(f"    - Markdown: {OUTPUT_MD_DIR.absolute()}")
    print(f"    - Chunks: {OUTPUT_CHUNKS_DIR.absolute()}")
    print("=" * 70)
    
    return stats["pdfs_failed"] == 0


if __name__ == "__main__":
    success = process_all_pdfs()
    sys.exit(0 if success else 1)
