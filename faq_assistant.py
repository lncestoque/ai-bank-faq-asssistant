# AI Banking FAQ Assistant (Streamlit App)

import streamlit as st
from transformers import pipeline

# --- Load Question-Answering Model ---
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- Load FAQ context from text file ---
with open("faq_context.txt", "r") as file:
    context = file.read()

# --- Streamlit App UI ---
st.set_page_config(page_title="AI Banking FAQ Assistant", page_icon=":bank:", layout="centered")
st.title("🤖 AI Banking FAQ Assistant")
st.markdown("Ask any question related to banking, loans, savings, or credit cards.")

user_question = st.text_input("💬 Your question:")

if user_question:
    result = qa_pipeline(question=user_question, context=context)
    st.markdown("### 📌 Answer:")
    if result['score'] > 0.2:
        st.success(result['answer'])
    else:
        st.warning("I'm not confident about that answer. Please try rephrasing your question.")

st.markdown("---")
st.caption("Powered by Hugging Face Transformers | Model: roberta-base-squad2")
