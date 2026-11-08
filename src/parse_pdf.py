import json
import re
from pathlib import Path

import pymupdf

# Build paths from this file's location so the script works from any folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw_pdfs"
OUT_DIR = PROJECT_DIR / "data" / "processed"

# A footer line like "12 / 46" holds the page number printed on the page
PRINTED_PAGE = re.compile(r"^\s*(\d+)\s*/\s*\d+\s*$", flags=re.MULTILINE)


def clean_text(text):
    """Tidy up spacing so the text is easier to work with."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)   # invisible control characters (e.g. \x07 bullets)
    text = re.sub(r"(\w)-\n(?=[A-Za-z]{2})", r"\1-", text)  # "pre-\nexisting" -> "pre-existing"
    text = re.sub(r"[ \t]+", " ", text)      # many spaces in a row -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # many blank lines -> one blank line
    return text.strip()                      # remove spaces at the start and end


def find_printed_page(text, pdf_page):
    """Return the page number printed on the page, or None if there isn't one.

    Looks for a footer like "12 / 46" first, then for a bare number such as "6"
    in the first or last few lines. A number only counts if it is close to the
    PDF page number, so table values like "100/ 200" or clause numbers like "21"
    aren't mistaken for it.
    """
    def close(n):
        return abs(n - pdf_page) <= 3

    for match in PRINTED_PAGE.finditer(text):
        if close(int(match.group(1))):
            return int(match.group(1))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:3] + lines[-3:]:
        if line.isdigit() and close(int(line)):
            return int(line)
    return None


def extract_pages(pdf_path):
    """Return a list like [{"page": 2, "printed_page": 1, "text": "..."}, ...] for one PDF.

    "page" is the position in the PDF file; "printed_page" is the number the
    reader sees on the page itself, which is what citations should show.
    """
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            text = clean_text(raw)
            if text:                         # skip empty pages
                pages.append({
                    "page": page_number,
                    "printed_page": find_printed_page(raw, page_number),
                    "text": text,
                })
        total = len(doc)
    # Some PDFs print no page numbers at all; then the PDF viewer's number is what readers use
    if sum(p["printed_page"] is not None for p in pages) < len(pages) / 2:
        for p in pages:
            p["printed_page"] = p["page"]
    if total and len(pages) < total / 2:
        print(f"Warning: {pdf_path.name} has text on only {len(pages)} of {total} pages. "
              "It may be a scanned PDF that needs OCR.")
    return pages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {RAW_DIR}")
    for pdf_path in pdf_paths:
        pages = extract_pages(pdf_path)
        out_path = OUT_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{pdf_path.name}: {len(pages)} pages saved to {out_path}")


if __name__ == "__main__":
    main()
