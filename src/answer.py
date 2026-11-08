import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieve import PROJECT_DIR, Retriever

load_dotenv(PROJECT_DIR / ".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You answer questions about health insurance policies.
Use ONLY the numbered policy excerpts provided. Do not use outside knowledge.
After every fact, cite its excerpt like [1] or [2][3].
If the excerpts don't contain the answer, reply exactly:
"The policy text provided doesn't cover this."
Keep answers short and in plain language. Quote exact numbers (days, months, amounts) as written."""


def answer(question, retriever, k=5):
    chunks = retriever.search(question, k=k)
    context = "\n\n".join(
        f"[{i}] ({c['plan']}, page {c['printed_page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, chunks


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the waiting period for pre-existing diseases?"
    text, chunks = answer(question, Retriever())
    print(text, "\n\nSources:")
    for i, c in enumerate(chunks, start=1):
        print(f"[{i}] page {c['printed_page']}")