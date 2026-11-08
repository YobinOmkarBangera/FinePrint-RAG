import csv
import json
import re
from collections import Counter
from pathlib import Path

# Build paths from this file's location so the script works from any folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SOURCES_CSV = PROJECT_DIR / "data" / "sources.csv"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE = 250     # words per chunk
OVERLAP = 50         # words shared by neighbouring chunks
REPEAT_SHARE = 0.5   # a line on more than half the pages counts as boilerplate
MIN_BOILERPLATE_LEN = 10   # shorter lines (like "i." or "(a)") are list markers, not headers
REQUIRED_COLUMNS = {"file_name", "insurer", "plan"}


def load_sources():
    """Read sources.csv into a dictionary keyed by file name."""
    # "utf-8-sig" also handles the hidden BOM that Excel adds when saving a CSV
    with SOURCES_CSV.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{SOURCES_CSV} needs a header row with columns: "
                             f"{', '.join(sorted(REQUIRED_COLUMNS))} (missing: {', '.join(sorted(missing))})")
        return {row["file_name"].strip(): row for row in reader}


def find_repeated_lines(pages):
    """Find short lines that appear on most pages (headers, footers, UIN codes)."""
    if len(pages) < 4:
        return set()
    counts = Counter()
    for page in pages:
        lines_on_page = {line.strip() for line in page["text"].splitlines() if line.strip()}
        counts.update(lines_on_page)
    limit = len(pages) * REPEAT_SHARE
    return {
        line for line, n in counts.items()
        if n > limit
        and MIN_BOILERPLATE_LEN <= len(line) < 120
    }


def remove_boilerplate(text, repeated):
    """Drop repeated lines and page-number lines, return clean text."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in repeated:
            continue
        if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", line, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return " ".join(kept)


def split_into_chunks(words, size=CHUNK_SIZE, overlap=OVERLAP):
    """Cut a list of words into overlapping windows."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def main():
    sources = load_sources()
    all_chunks = []

    for json_path in sorted(PROCESSED_DIR.glob("*.json")):
        pdf_name = json_path.stem + ".pdf"
        if pdf_name not in sources:
            print(f"Skipping {json_path.name}: add {pdf_name} to sources.csv")
            continue
        info = sources[pdf_name]
        pages = json.loads(json_path.read_text(encoding="utf-8"))
        repeated = find_repeated_lines(pages)

        for page in pages:
            clean = remove_boilerplate(page["text"], repeated)
            for i, text in enumerate(split_into_chunks(clean.split())):
                all_chunks.append({
                    "chunk_id": f"{json_path.stem}-p{page['page']}-c{i}",
                    "insurer": info["insurer"],
                    "plan": info["plan"],
                    "page": page["page"],                       # position in the PDF file
                    "printed_page": page.get("printed_page"),   # number shown on the page (use for citations)
                    "text": text,
                })

        print(f"{json_path.name}: removed {len(repeated)} repeated lines")

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if not all_chunks:
        print(f"No chunks were made. Check that {SOURCES_CSV.name} lists your PDFs.")
    else:
        avg = sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks)
        print(f"Saved {len(all_chunks)} chunks to {CHUNKS_FILE} (average {avg:.0f} words)")


if __name__ == "__main__":
    main()