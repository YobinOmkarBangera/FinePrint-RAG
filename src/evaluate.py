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


def load_questions(known_plans):
    with EVAL_FILE.open(encoding="utf-8-sig") as f:
        rows = []
        # start=2 because line 1 of the file is the header row
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            if not row["question"].strip():
                continue
            plan = row["plan"].strip()
            if plan not in known_plans:
                raise SystemExit(
                    f'{EVAL_FILE.name} line {line_number}: unknown plan "{plan}". '
                    f"Use one of: {', '.join(known_plans)}"
                )
            rows.append((row["question"].strip(), plan, parse_pages(row["expected_page"], line_number)))
        return rows


def is_hit(chunks, plan, expected):
    """A hit if any retrieved chunk is from the right plan AND on a correct page."""
    return any(c["plan"] == plan and c["printed_page"] in expected for c in chunks)


def main(k=5):
    retriever = Retriever()
    questions = load_questions(retriever.plans)
    answerable = [q for q in questions if q[2]]
    if not answerable:
        raise SystemExit(f"No questions with an expected_page in {EVAL_FILE.name}")

    hits_filtered = hits_all = 0
    print(f"{'one policy':<12}{'compare':<11}question")
    for question, plan, expected in answerable:
        # 1. Like the app with one policy picked in the sidebar: top k from that policy
        filtered = is_hit(retriever.search(question, k=k, plan=plan), plan, expected)
        # 2. Like the app on "All policies (compare)": top 3 from every policy
        everywhere = is_hit(retriever.search_each_plan(question, k=3), plan, expected)
        hits_filtered += filtered
        hits_all += everywhere
        print(f"{'HIT' if filtered else 'MISS':<12}{'HIT' if everywhere else 'MISS':<11}[{plan}] {question}")

    n = len(answerable)
    print(f"\nOne policy  (top {k} from the chosen policy):  {hits_filtered}/{n} = {hits_filtered / n:.0%}")
    print(f"Compare mode (top 3 from each policy):        {hits_all}/{n} = {hits_all / n:.0%}")
    skipped = len(questions) - n
    if skipped:
        print(f"({skipped} questions with no expected page were skipped; use them to test answer.py)")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
