import streamlit as st

from helpdesk_agent import (
    check_gemini,
    build_embedding_index,
    run_agent,
    TICKETS,
    POLICIES,
    GEN_MODEL,
    EMBED_MODEL,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Grounded Student Helpdesk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .status-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .ticket-card {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZATION
# ============================================================

@st.cache_resource
def initialize_system():

    check_gemini()

    build_embedding_index()

    return True


try:

    initialize_system()

    system_ready = True

except Exception as error:

    system_ready = False

    st.error(
        f"System initialization failed:\n\n{error}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Helpdesk")

    st.caption(
        "Grounded Student AI Assistant"
    )

    st.divider()

    st.subheader("⚙️ System")

    st.success(
        "🟢 Gemini API Connected"
    )

    st.success(
        "🟢 Knowledge Base Ready"
    )

    st.divider()

    st.subheader("🤖 AI Configuration")

    st.write(
        f"**Generation Model**  \n"
        f"`{GEN_MODEL}`"
    )

    st.write(
        f"**Embedding Model**  \n"
        f"`{EMBED_MODEL}`"
    )

    st.write(
        "**Retrieval**  \n"
        "`Cosine Similarity`"
    )

    st.write(
        "**Architecture**  \n"
        "`RAG + Agentic Tool Calling`"
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    st.metric(
        "Policies",
        len(POLICIES)
    )

    st.divider()

    st.subheader("💡 Example Questions")

    examples = [
        "What attendance percentage do I need for FAT?",
        "When can I request revaluation?",
        "What are the library timings?",
        "What is the assignment submission policy?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state[
                "selected_question"
            ] = example


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎓 Grounded Student Helpdesk'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered university helpdesk using '
    'Gemini, RAG, embeddings, and agentic tools.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TOP STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📚 Policies",
        len(POLICIES),
    )

with col2:

    st.metric(
        "🤖 AI Model",
        "Gemini 3.8",
    )

with col3:

    st.metric(
        "🎫 Support Tickets",
        len(TICKETS),
    )


st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# QUESTION INPUT
# ============================================================

selected_question = st.session_state.pop(
    "selected_question",
    None,
)

prompt = st.chat_input(
    "Ask a question about university policies..."
)

if selected_question:

    prompt = selected_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    if not system_ready:

        st.error(
            "The helpdesk is not ready. "
            "Please refresh the application."
        )

        st.stop()

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching knowledge base..."
        ):

            try:

                answer = run_agent(
                    prompt,
                    verbose=False,
                )

            except Exception as error:

                answer = (
                    "Sorry, an error occurred "
                    "while processing your question.\n\n"
                    f"Technical details: {error}"
                )

        st.markdown(answer)

    # --------------------------------------------------------
    # SAVE RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# SUPPORT TICKETS
# ============================================================

if TICKETS:

    st.divider()

    st.subheader(
        "🎫 Support Tickets"
    )

    for ticket in reversed(TICKETS):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### Ticket #{ticket['id']}"
            )

            st.write(
                f"**Question:** "
                f"{ticket['question']}"
            )

            st.write(
                f"**Reason:** "
                f"{ticket['reason']}"
            )

            st.caption(
                f"Created: "
                f"{ticket['created_at']}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Grounded Student Helpdesk | "
    "Gemini • Embeddings • RAG • Agentic AI"
)
