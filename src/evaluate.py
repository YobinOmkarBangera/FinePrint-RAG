import csv
import sys

from retrieve import PROJECT_DIR, Retriever

EVAL_FILE = PROJECT_DIR / "data" / "eval_questions.csv"


def parse_pages(value, line_number):
    """Turn "13|30" into {13, 30}. An empty value means the policy doesn't answer it."""
    value = value.strip()
    if not value:
        return set()
    try:
        return {int(part) for part in value.split("|")}
    except ValueError:
        raise SystemExit(
            f"{EVAL_FILE.name} line {line_number}: expected_page must be a page number "
            f'like 26, or several like 13|30, not "{value}"'
        )


def load_questions():
    with EVAL_FILE.open(encoding="utf-8-sig") as f:
        # start=2 because line 1 of the file is the header row
        return [
            (row["question"].strip(), parse_pages(row["expected_page"], line_number))
            for line_number, row in enumerate(csv.DictReader(f), start=2)
            if row["question"].strip()
        ]


def main(k=5):
    questions = load_questions()
    answerable = [(q, pages) for q, pages in questions if pages]
    if not answerable:
        raise SystemExit(f"No questions with an expected_page in {EVAL_FILE.name}")

    retriever = Retriever()
    hits = 0
    for question, expected in answerable:
        got = [c["printed_page"] for c in retriever.search(question, k=k)]
        found = bool(expected & set(got))   # a hit if ANY correct page was retrieved
        hits += found
        expected_text = "|".join(str(p) for p in sorted(expected))
        print(f"{'HIT ' if found else 'MISS'}  expected p{expected_text:<6} got {got}  {question}")

    skipped = len(questions) - len(answerable)
    print(f"\nHit rate @{k}: {hits}/{len(answerable)} = {hits / len(answerable):.0%}")
    if skipped:
        print(f"({skipped} questions with no expected page were skipped; use them to test answer.py)")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
