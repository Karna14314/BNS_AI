import os
import re
import fitz  # PyMuPDF
import pymupdf4llm

def extract_pdf_to_md(pdf_files, output_dir="extracted_md"):
    """Extracts PDFs to Markdown, handling page errors gracefully."""
    os.makedirs(output_dir, exist_ok=True)
    md_files = []
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Warning: File not found: '{pdf_path}' in {os.getcwd()}")
            continue
            
        print(f"Processing PDF: {pdf_path}")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_md_path = os.path.join(output_dir, f"{base_name}.md")
        
        try:
            # Try converting the whole document at once
            print("  -> Attempting full document conversion...")
            md_text = pymupdf4llm.to_markdown(pdf_path)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"  -> Successfully converted to {output_md_path}")
            md_files.append(output_md_path)
            
        except Exception as e:
            print(f"  -> Error converting whole document {pdf_path}: {e}")
            print("  -> Attempting page-by-page extraction fallback...")
            
            md_text_full = ""
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    try:
                        md_text = pymupdf4llm.to_markdown(doc, pages=[page_num])
                        md_text_full += md_text + "\n\n"
                    except Exception as page_e:
                        print(f"    -> Warning: Could not read page {page_num + 1} of {pdf_path}. Skipping page. Error: {page_e}")
                        
                with open(output_md_path, "w", encoding="utf-8") as f:
                    f.write(md_text_full)
                print(f"  -> Successfully converted (page-by-page) to {output_md_path}")
                md_files.append(output_md_path)
            except Exception as doc_e:
                 print(f"  -> Critical Error opening document {pdf_path}: {doc_e}")
                 
    return md_files

def chunk_markdown_files(md_files, chunk_dir="chunks"):
    """Splits Markdown files into chunks based on ## or ### headers."""
    os.makedirs(chunk_dir, exist_ok=True)
    
    for md_path in md_files:
        print(f"Chunking: {md_path}")
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        
        # Name chunks like {base_name}_chunk_001.txt
        file_chunk_counter = 1
        
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_chunk_lines = []
        
        for line in lines:
            # Check for major headers: starts with '## ' or '### '
            if line.startswith("## ") or line.startswith("### "):
                # If we already have content in the current chunk, save it
                if "".join(current_chunk_lines).strip():
                    chunk_filename = f"{base_name}_chunk_{file_chunk_counter:03d}.txt"
                    chunk_path = os.path.join(chunk_dir, chunk_filename)
                    with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                        chunk_file.write("".join(current_chunk_lines))
                    file_chunk_counter += 1
                
                # Start a new chunk with the header
                current_chunk_lines = [line]
            else:
                # Append to current chunk
                current_chunk_lines.append(line)
                
        # Save the last chunk if there's remaining content
        if "".join(current_chunk_lines).strip():
            chunk_filename = f"{base_name}_chunk_{file_chunk_counter:03d}.txt"
            chunk_path = os.path.join(chunk_dir, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                chunk_file.write("".join(current_chunk_lines))
            file_chunk_counter += 1
            
        print(f"  -> Created {file_chunk_counter - 1} chunks for {base_name}")

if __name__ == "__main__":
    pdf_list = [
        "BNS Book_After Correction.pdf",
        "Final_BNSS.pdf",
        "Primar on New Criminal Laws.pdf"
    ]
    
    print("=== Phase 1: Extracting PDFs to Markdown ===")
    extracted_md_files = extract_pdf_to_md(pdf_list)
    
    if not extracted_md_files:
        print("\nNo markdown files were generated. Exiting.")
        exit(1)
        
    print("\n=== Phase 2: Chunking Markdown Files ===")
    chunk_markdown_files(extracted_md_files)
    
    print("\nProcess completed successfully!")
