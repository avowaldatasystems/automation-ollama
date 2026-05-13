import streamlit as st
from graph import app

st.title("PDF Chatbot")

# Form enables Enter key submission
with st.form("chat_form"):

    question = st.text_input(
        "Ask a question"
    )

    submitted = st.form_submit_button("Ask")

if submitted and question:

    result = app.invoke({
        "question": question
    })

    st.write(result["answer"])