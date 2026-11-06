import json
import re
from pathlib import Path

import fitz  # PyMuPDF

RAW_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data/processed")


def clean_text(text):
    """Tidy up spacing so the text is easier to work with."""
    text = re.sub(r"[ \t]+", " ", text)      # many spaces in a row -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # many blank lines -> one blank line
    return text.strip()                      # remove spaces at the start and end


def extract_pages(pdf_path):
    """Return a list like [{"page": 1, "text": "..."}, ...] for one PDF."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:                         # skip empty pages
                pages.append({"page": page_number, "text": text})
    return pages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        pages = extract_pages(pdf_path)
        out_path = OUT_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{pdf_path.name}: {len(pages)} pages saved to {out_path}")


if __name__ == "__main__":
    main()