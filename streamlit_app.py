import streamlit as st

from helpdesk_agent import (
    check_gemini,
    build_embedding_index,
    run_agent,
    TICKETS,
    POLICIES,
    GEN_MODEL,
    EMBED_MODEL,
    LAST_RETRIEVAL,
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

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SYSTEM
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

    if system_ready:

        st.success(
            "🟢 Gemini API Connected"
        )

        st.success(
            "🟢 Knowledge Base Ready"
        )

    else:

        st.error(
            "🔴 System Not Ready"
        )

    st.divider()

    st.subheader("🤖 AI Configuration")

    st.write(
        f"**Generation Model**\n\n"
        f"`{GEN_MODEL}`"
    )

    st.write(
        f"**Embedding Model**\n\n"
        f"`{EMBED_MODEL}`"
    )

    st.write(
        "**Retrieval**\n\n"
        "`Cosine Similarity`"
    )

    st.write(
        "**Architecture**\n\n"
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
# TOP METRICS
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
    # RESET RETRIEVAL STATUS
    # --------------------------------------------------------

    LAST_RETRIEVAL["found"] = False
    LAST_RETRIEVAL["title"] = None
    LAST_RETRIEVAL["score"] = 0.0

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

                error_text = str(
                    error
                ).upper()

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    answer = (
                        "⚠️ **AI service quota temporarily "
                        "exceeded.**\n\n"
                        "The Gemini API free-tier quota has "
                        "been reached. Please try again later."
                    )

                elif (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    answer = (
                        "⚠️ **AI service is temporarily busy.**\n\n"
                        "Please try again in a few moments."
                    )

                else:

                    answer = (
                        "⚠️ **Something went wrong while "
                        "processing your question.**\n\n"
                        "Please try again later."
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

    # ========================================================
    # RAG TRANSPARENCY
    # ========================================================

    retrieval = LAST_RETRIEVAL

    with st.expander(
        "🔎 RAG Retrieval Details",
        expanded=False,
    ):

        if retrieval["found"]:

            st.success(
                "✅ Confident policy match found"
            )

            st.write(
                f"**Retrieved Policy:** "
                f"{retrieval['title']}"
            )

            st.write(
                f"**Cosine Similarity:** "
                f"{retrieval['score']:.4f}"
            )

            st.progress(
                min(
                    max(
                        retrieval["score"],
                        0.0
                    ),
                    1.0
                )
            )

            st.caption(
                "The policy was retrieved from the "
                "university knowledge base before "
                "the final answer was generated."
            )

        else:

            if retrieval["score"] > 0:

                st.warning(
                    "⚠️ No confident policy match found"
                )

                st.write(
                    f"**Best Similarity Score:** "
                    f"{retrieval['score']:.4f}"
                )

                st.caption(
                    "No sufficiently relevant policy "
                    "was found in the knowledge base."
                )

            else:

                st.info(
                    "ℹ️ No retrieval information available."
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
