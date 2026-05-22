import streamlit as st

from graph import graph


st.set_page_config(
    page_title="Employee DB Agent",
    layout="wide"
)

st.title(
    "👨‍💼 Employee Database Agent"
)

st.caption(
    "Ask questions about employees"
)


# -------------------------
# Session Memory
# -------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# -------------------------
# Show Chat History
# -------------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# -------------------------
# Chat Input
# -------------------------
question = st.chat_input(
    "Ask something..."
)

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role":
            "user",

            "content":
            question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(question)

    # Graph call
    result = graph.invoke(
        {
            "question":
            question,

            "chat_history":
            st.session_state.chat_history
        }
    )

    answer = result["response"]

    # Save assistant message
    st.session_state.messages.append(
        {
            "role":
            "assistant",

            "content":
            answer
        }
    )

    # SAVE MEMORY
    st.session_state.chat_history = (
        result["chat_history"]
    )

    with st.chat_message(
        "assistant"
    ):

        st.write(answer)