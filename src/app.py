import streamlit as st

from answer import answer
from retrieve import Retriever

st.set_page_config(page_title="FinePrint", page_icon="📄")
st.title("FinePrint")
st.caption("Ask about your health insurance policy. Every answer cites the policy clause.")


@st.cache_resource  # load the model and index once, not on every question
def get_retriever():
    return Retriever()


question = st.text_input("Your question", placeholder="Is cataract surgery covered?")

if question:
    with st.spinner("Reading the fine print..."):
        text, chunks = answer(question, get_retriever())
    st.markdown(text)

    with st.expander("Sources"):
        for i, c in enumerate(chunks, start=1):
            st.markdown(f"**[{i}] {c['plan']}, page {c['printed_page']}**")
            st.write(c["text"])