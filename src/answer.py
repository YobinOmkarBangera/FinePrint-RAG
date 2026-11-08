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


def build_messages(question, chunks):
    context = "\n\n".join(
        f"[{i}] ({c['plan']}, page {c['printed_page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


def answer(question, retriever, k=5, plan=None):
    """Return (answer_text, chunks) in one go."""
    chunks = retriever.search(question, k=k, plan=plan)
    response = client.chat.completions.create(
        model=MODEL, temperature=0, messages=build_messages(question, chunks)
    )
    return clean_citations(response.choices[0].message.content), chunks


def stream_answer(question, retriever, k=5, plan=None):
    """Return (chunks, generator); the generator yields the answer piece by piece."""
    chunks = retriever.search(question, k=k, plan=plan)
    stream = client.chat.completions.create(
        model=MODEL, temperature=0, messages=build_messages(question, chunks), stream=True
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
        print(f"[{i}] page {c['printed_page']}")