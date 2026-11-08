import html
import re
import time

import pymupdf
import streamlit as st

from answer import NOT_COVERED, clean_citations, stream_answer
from retrieve import PROJECT_DIR, QUERY_PREFIX, Retriever

RAW_DIR = PROJECT_DIR / "data" / "raw_pdfs"
ALL_POLICIES = "All policies (compare)"
EXAMPLES = [  # (button label, full question)
    ("⏳ Pre-existing diseases", "What is the waiting period for pre-existing diseases?"),
    ("🏥 Room rent limit", "What is the room rent limit?"),
    ("👶 Maternity cover", "Are maternity expenses covered?"),
    ("🐶 Pet cover?", "Is my pet dog covered under this policy?"),
]

st.set_page_config(page_title="FinePrint", page_icon="📄", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
  border-radius: 20px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 12px 30px -12px rgba(79, 70, 229, .55);
}
.hero h1 { color: #fff; font-size: 2.6rem; margin: 0; letter-spacing: -0.02em; }
.hero p { margin: .4rem 0 0; font-size: 1.08rem; opacity: .92; }
.hero .chips { margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.hero .chip {
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
  border-radius: 999px; padding: .25rem .8rem; font-size: .85rem;
}
.cite {
  display: inline-block; min-width: 1.35rem; padding: 0 .35rem; margin: 0 .1rem;
  border-radius: 6px; background: #EEF2FF; color: #4338CA; font-weight: 700;
  font-size: .78rem; text-align: center; border: 1px solid #C7D2FE;
}
.source-card {
  background: #fff; border: 1px solid #E6E6F0; border-left: 4px solid #4F46E5;
  border-radius: 12px; padding: .8rem 1rem; margin-bottom: .2rem;
}
.source-card.uncited { border-left-color: #CBD5E1; }
.source-card .meta { font-size: .82rem; color: #6B7280; margin-bottom: .35rem; }
.source-card .meta b { color: #1E1B3A; }
.source-card .snippet { font-size: .92rem; color: #374151; line-height: 1.5; }
.page-pill {
  background: #1E1B3A; color: #fff; border-radius: 999px; padding: .05rem .55rem;
  font-size: .75rem; font-weight: 600; margin-left: .3rem;
}
.not-covered {
  background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px;
  padding: .8rem 1rem; color: #92400E;
}
.timing { font-size: .8rem; color: #6B7280; margin-top: .4rem; }
.step { display: flex; gap: .6rem; margin-bottom: .55rem; font-size: .9rem; }
.step .num {
  flex: none; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #EEF2FF;
  color: #4338CA; font-weight: 700; font-size: .8rem; display: grid; place-items: center;
}
.disclaimer { font-size: .78rem; color: #6B7280; margin-top: 2rem; text-align: center; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading the policy index...")
def get_retriever():
    return Retriever()


def pdf_path_for(chunk):
    # chunk_id looks like "<pdf stem>-p<page>-c<n>"
    return RAW_DIR / f"{chunk['chunk_id'].rsplit('-', 2)[0]}.pdf"


def normalise(word):
    return re.sub(r"\W", "", word.lower())


@st.cache_data(show_spinner=False)
def render_page(pdf_path, page_number, chunk_text):
    """Render one PDF page as a PNG with the retrieved clause highlighted."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]
        words = page.get_text("words")          # (x0, y0, x1, y1, word, ...)
        page_words = [normalise(w[4]) for w in words]
        chunk_words = [w for w in (normalise(x) for x in chunk_text.split()) if w]

        def find(seq, start=0):
            n = len(seq)
            for i in range(start, len(page_words) - n + 1):
                if page_words[i:i + n] == seq:
                    return i
            return None

        start = find(chunk_words[:4])
        end = find(chunk_words[-4:], start or 0) if start is not None else None
        if start is not None:
            last = (end + 3) if end is not None else min(start + len(chunk_words), len(words) - 1)
            for w in words[start:last + 1]:
                page.add_highlight_annot(pymupdf.Rect(w[:4]))
        return page.get_pixmap(dpi=110).tobytes("png")


# ---------- Rendering helpers ----------
def with_citation_badges(text):
    """Escape the model's text, then turn [1] into a styled badge."""
    safe = html.escape(text, quote=False)
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', safe)


def show_answer(text):
    if text.strip().strip('"') == NOT_COVERED:
        st.markdown(
            '<div class="not-covered">🔎 <b>Not found in the policy.</b> '
            "None of the retrieved clauses answer this, so FinePrint won't guess. "
            "Try rephrasing, or check with your insurer.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(with_citation_badges(text), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def best_snippet(chunk_text, question, length=320):
    """Show the part of a chunk whose MEANING is closest to the question.

    Uses the same embedding model as retrieval, so "waiting period for
    pre-existing diseases" lands on the PED clause, not on another clause that
    merely repeats the words "waiting period".
    """
    sentences = [s for s in re.split(r"(?<=[.:;])\s+", chunk_text) if s.strip()] or [chunk_text]
    model = get_retriever().model
    q = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    s = model.encode(sentences, normalize_embeddings=True)
    start = int((s @ q).argmax())
    snippet = " ".join(sentences[start:])
    prefix = "…" if start else ""
    suffix = "…" if len(snippet) > length else ""
    return prefix + snippet[:length] + suffix


def source_card(i, chunk, question, text, cited):
    snippet = html.escape(best_snippet(chunk["text"], question))
    page_label = chunk["printed_page"] or "cover"
    st.markdown(
        f'<div class="source-card {"" if cited else "uncited"}">'
        f'<div class="meta"><span class="cite">{i}</span> <b>{html.escape(chunk["plan"])}</b>'
        f'<span class="page-pill">page {page_label}</span></div>'
        f'<div class="snippet">{snippet}</div></div>',
        unsafe_allow_html=True,
    )


def show_sources(question, text, chunks):
    numbers = range(1, len(chunks) + 1)
    cited = [i for i in numbers if f"[{i}]" in text]
    plans = list(dict.fromkeys(c["plan"] for c in chunks))   # unique, in retrieval order

    if len(plans) == 1:
        if cited:
            st.markdown(f"**Sources** · {len(cited)} of {len(chunks)} retrieved clauses cited")
        source_list(question, text, chunks, list(numbers), cited)
        return

    # Several policies: one tab of sources per policy
    st.markdown(f"**Sources** · {len(cited)} of {len(chunks)} retrieved clauses cited, "
                f"across {len(plans)} policies")
    for tab, plan in zip(st.tabs(plans), plans):
        with tab:
            mine = [i for i in numbers if chunks[i - 1]["plan"] == plan]
            source_list(question, text, chunks, mine, [i for i in mine if i in cited])


def source_list(question, text, chunks, numbers, cited):
    uncited = [i for i in numbers if i not in cited]
    for i in cited:
        c = chunks[i - 1]
        source_card(i, c, question, text, cited=True)
        pdf_path = pdf_path_for(c)
        if pdf_path.exists():
            with st.expander(f"View clause [{i}] on the original page"):
                st.image(render_page(str(pdf_path), c["page"], c["text"]), width="stretch")

    if uncited:
        with st.expander(f"Other clauses checked ({len(uncited)})"):
            for i in uncited:
                source_card(i, chunks[i - 1], question, text, cited=False)


def show_timing(t):
    st.markdown(
        f'<div class="timing">⚡ Retrieved in {t["retrieve"]:.2f}s · '
        f'answered in {t["generate"]:.1f}s · searched {t["scope"]}</div>',
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
retriever = get_retriever()
with st.sidebar:
    st.markdown("### 📚 Policy library")
    plan = st.selectbox("Search in", [ALL_POLICIES] + retriever.plans)
    if plan == ALL_POLICIES and len(retriever.plans) > 1:
        st.caption("Each policy is answered separately, so you can compare them side by side.")
    pages = {(c["chunk_id"].rsplit("-", 2)[0], c["page"]) for c in retriever.chunks}
    a, b, c = st.columns(3)
    a.metric("Policies", len(retriever.plans))
    b.metric("Pages", len(pages))
    c.metric("Clauses", len(retriever.chunks))

    st.markdown("### ⚙️ How it works")
    for n, step in enumerate([
        "Policy PDFs are parsed page by page, and headers and footers are stripped.",
        "Text is split into overlapping clauses and embedded with <b>bge-small</b>.",
        "Your question runs through <b>hybrid search</b>: vector search + BM25, fused by rank.",
        "An LLM answers <b>only from the retrieved clauses</b> and cites each one.",
    ], start=1):
        st.markdown(f'<div class="step"><div class="num">{n}</div><div>{step}</div></div>',
                    unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <h1>📄 FinePrint</h1>
  <p>Ask anything about your health insurance policy. Every answer is backed by the exact clause and page.</p>
  <div class="chips">
    <span class="chip">🔍 Hybrid retrieval</span>
    <span class="chip">📌 Page-level citations</span>
    <span class="chip">🛡️ Says so when the policy is silent</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# Example questions, shown until the first question is asked
pending = None
if not st.session_state.history:
    st.markdown("##### Try one of these")
    cols = st.columns(len(EXAMPLES))
    for col, (label, q) in zip(cols, EXAMPLES):
        if col.button(label, help=q, width="stretch"):
            pending = q

# ---------- Conversation ----------
for turn in st.session_state.history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(html.escape(turn["question"]))
    with st.chat_message("assistant", avatar="📄"):
        show_answer(turn["answer"])
        show_timing(turn["timing"])
        show_sources(turn["question"], turn["answer"], turn["chunks"])

question = st.chat_input("Ask about waiting periods, exclusions, limits…") or pending

if question:
    selected = None if plan == ALL_POLICIES else plan
    with st.chat_message("user", avatar="🧑"):
        st.markdown(html.escape(question))
    with st.chat_message("assistant", avatar="📄"):
        t0 = time.perf_counter()
        with st.spinner("Searching the fine print..."):
            chunks, pieces = stream_answer(question, retriever, plan=selected)
        t1 = time.perf_counter()
        if not chunks:
            st.warning("No clauses found for this policy.")
            st.stop()
        placeholder = st.empty()
        text = clean_citations(placeholder.write_stream(pieces))
        timing = {
            "retrieve": t1 - t0,
            "generate": time.perf_counter() - t1,
            "scope": f"all {len(retriever.plans)} policies" if selected is None else selected,
        }
        with placeholder.container():
            show_answer(text)
        show_timing(timing)
        show_sources(question, text, chunks)
    st.session_state.history.append(
        {"question": question, "answer": text, "chunks": chunks, "timing": timing}
    )
    st.rerun()  # redraw from history so the example buttons disappear

st.markdown(
    '<div class="disclaimer">FinePrint explains policy wording and is not legal or financial advice. '
    "Always confirm coverage with your insurer.</div>",
    unsafe_allow_html=True,
)
