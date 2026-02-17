
import os
import re
import sys

# Try imports for PDF handling, but don't crash if missing
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

def extract_links_from_text(text):
    links = []
    # Find URLs (http/https)
    # Regex captures http/s :// followed by non-whitespace
    urls = re.findall(r'(https?://[^\s)\]"\']+)', text)
    for url in urls:
        # Filter for specific domains of interest
        if any(domain in url for domain in ['arxiv.org', 'github.com', 'cvpr', 'openaccess.thecvf.com']):
            links.append(url)
    return links

def extract_links_from_pdf(pdf_path):
    if not HAS_PYPDF2:
        print(f"Warning: PyPDF2 not installed. Skipping PDF extraction for {pdf_path}")
        return []
    
    links = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            links.extend(extract_links_from_text(text))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return links

def extract_links_from_md(md_path):
    links = []
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
            links.extend(extract_links_from_text(text))
    except Exception as e:
        print(f"Error reading Markdown {md_path}: {e}")
    return links

def main():
    reports_dir = os.path.join("Research_lore", "llm_reports")
    if not os.path.exists(reports_dir):
        print(f"Directory not found: {reports_dir}")
        return

    print(f"Scanning files in {reports_dir}...")
    files = [f for f in os.listdir(reports_dir) if os.path.isfile(os.path.join(reports_dir, f))]
    
    if not files:
        print("No files found.")
        return

    all_links = set()

    for filename in sorted(files):
        file_path = os.path.join(reports_dir, filename)
        file_links = []
        
        if filename.lower().endswith(".pdf"):
            print(f"\nProcessing PDF: {filename}")
            file_links = extract_links_from_pdf(file_path)
        elif filename.lower().endswith(".md"):
            print(f"\nProcessing Markdown: {filename}")
            file_links = extract_links_from_md(file_path)
        else:
            continue
            
        unique_file_links = sorted(list(set(file_links)))
        for link in unique_file_links:
            print(f"  - {link}")
            all_links.add(link)

    print(f"\nTotal unique target links found: {len(all_links)}")

if __name__ == "__main__":
    main()
