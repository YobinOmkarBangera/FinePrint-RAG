import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import PROJECT_DIR, Retriever

load_dotenv(PROJECT_DIR / ".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

NOT_COVERED = "The policy text provided doesn't cover this."

SYSTEM_PROMPT = f"""You answer questions about health insurance policies.
Use ONLY the numbered policy excerpts provided. Do not use outside knowledge.
After every fact, cite its excerpt number in plain square brackets, like [1] or [2][3].
Never use any other citation style.
If the excerpts don't contain the answer, reply exactly:
"{NOT_COVERED}"
Keep answers short and in plain language. Quote exact numbers (days, months, amounts) as written."""


def clean_citations(text):
    """Turn the model's occasional native style "【1†L1-L3】" into "[1]"."""
    return re.sub(r"【(\d+)[^】]*】", r"[\1]", text)


PER_POLICY_INSTRUCTIONS = f"""The excerpts come from several different policies.
Answer separately for EACH policy, in the order given, under a heading "### <policy name>".
Use only that policy's excerpts under its heading; never mix facts between policies.
If a policy's excerpts don't answer the question, write "{NOT_COVERED}" under its heading.
Finish with one short line starting "**In short:**" that compares the policies."""


def retrieve(question, retriever, k=5, plan=None):
    """Return (chunks, per_policy). With no plan chosen, take the best clauses from each policy."""
    if plan is None and len(retriever.plans) > 1:
        return retriever.search_each_plan(question, k=3), True
    return retriever.search(question, k=k, plan=plan), False


def build_messages(question, chunks, per_policy=False):
    parts, current_plan = [], None
    for i, c in enumerate(chunks, start=1):
        if per_policy and c["plan"] != current_plan:
            current_plan = c["plan"]
            parts.append(f"=== Policy: {current_plan} ===")
        parts.append(f"[{i}] ({c['plan']}, page {c['printed_page']})\n{c['text']}")
    context = "\n\n".join(parts)
    instructions = f"\n\n{PER_POLICY_INSTRUCTIONS}" if per_policy else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}{instructions}"},
    ]


def answer(question, retriever, k=5, plan=None):
    """Return (answer_text, chunks) in one go."""
    chunks, per_policy = retrieve(question, retriever, k=k, plan=plan)
    response = client.chat.completions.create(
        model=MODEL, temperature=0, messages=build_messages(question, chunks, per_policy)
    )
    return clean_citations(response.choices[0].message.content), chunks


def stream_answer(question, retriever, k=5, plan=None):
    """Return (chunks, generator); the generator yields the answer piece by piece."""
    chunks, per_policy = retrieve(question, retriever, k=k, plan=plan)
    stream = client.chat.completions.create(
        model=MODEL, temperature=0, messages=build_messages(question, chunks, per_policy), stream=True
    )

    def pieces():
        for part in stream:
            text = part.choices[0].delta.content
            if text:
                yield text

    return chunks, pieces()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the waiting period for pre-existing diseases?"
    text, chunks = answer(question, Retriever())
    print(text, "\n\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] {c['plan']}, page {c['printed_page']}")