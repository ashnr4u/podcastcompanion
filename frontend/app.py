import sys
from pathlib import Path
import uuid

import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_SRC = BASE_DIR / "backend" / "src"

sys.path.insert(0, str(BACKEND_SRC))


# ============================================================
# BACKEND
# ============================================================

from graph import app_graph
import retrieval


@st.cache_resource
def initialize_retrieval():

    retrieval.initialize()

    return retrieval


# Initialize retrieval resources when Streamlit starts
initialize_retrieval()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fermi Podcast Companion",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎙️ Fermi Podcast Companion")

    st.caption(
        "Ask questions across the supplied research-paper episodes."
    )

    st.divider()

    st.subheader("Episodes")

    episodes = [
        "Einstein — Special Relativity",
        "Hawking — Black Hole Radiation",
        "Watson & Crick — The Double Helix",
        "Shannon — Birth of Information",
        "Vaswani et al. — Attention Is All You Need",
    ]

    for episode in episodes:
        st.write(f"• {episode}")

    st.divider()

    st.caption(
        "Answers are grounded only in the supplied podcast transcripts."
    )

    if st.button(
        "New conversation",
        use_container_width=True
    ):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("Fermi Podcast Companion")

st.markdown(
    "Explore the ideas discussed across the five supplied episodes. "
    "Ask questions, compare papers, or follow up naturally."
)


# ============================================================
# DISPLAY UI CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about the episodes..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------------------------
    # LangGraph thread configuration
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    try:

        # ----------------------------------------------------
        # Invoke LangGraph
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = app_graph.invoke(
                    {
                        "question": question
                    },
                    config=config
                )

            answer = result.get(
                "answer",
                "I couldn't find enough information in the provided episodes."
            )

            st.markdown(answer)

        # ----------------------------------------------------
        # Save UI history
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        error_message = (
            "Something went wrong while processing your question."
        )

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message
            }
        )

        print(f"[FRONTEND ERROR] {type(e).__name__}: {e}")


# ============================================================
# OPTIONAL MEMORY DEBUG
# ============================================================

# Uncomment this section temporarily if you want to verify
# that LangGraph MemorySaver is actually storing the state.

with st.sidebar.expander("Debug Memory"):

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    saved_state = app_graph.get_state(config)

    st.write("Thread ID:")
    st.code(st.session_state.thread_id)

    st.write("Saved chat history:")
    st.write(
        saved_state.values.get(
            "chat_history",
            []
        )
    )
