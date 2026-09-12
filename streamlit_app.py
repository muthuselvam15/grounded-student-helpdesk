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
    layout="centered",
)


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

st.title("🎓 Grounded Student Helpdesk")

st.caption(
    "Gemini + RAG + Embeddings + Agentic Tool Calling"
)


# ============================================================
# INITIALIZE GEMINI AND EMBEDDINGS
# ============================================================

if "initialized" not in st.session_state:

    with st.spinner(
        "Connecting to Gemini and loading the policy knowledge base..."
    ):

        try:

            # Check Gemini API
            check_gemini()

            # Build policy embeddings
            build_embedding_index()

            st.session_state.initialized = True

        except Exception as error:

            st.error(
                f"❌ Initialization failed:\n\n{error}"
            )

            st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        f"**Generation Model:** `{GEN_MODEL}`"
    )

    st.write(
        f"**Embedding Model:** `{EMBED_MODEL}`"
    )

    st.write(
        "**Retrieval:** Cosine Similarity"
    )

    st.write(
        "**Architecture:** RAG + Agentic Tool Calling"
    )

    st.write(
        f"**Policies:** {len(POLICIES)}"
    )

    st.divider()

    st.subheader("🔧 System Status")

    st.success("Gemini API Connected")

    st.success("Gemini Models Available")

    st.success("Knowledge Base Ready")


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.markdown(
    """
This AI helpdesk answers student questions using a **grounded
policy knowledge base**.

### How it works

**Student Question → Gemini → Policy Search → Gemini Embeddings →
Cosine Similarity → Grounded Answer**

If no confident policy match is found:

**Student Question → Search → No Match → Support Ticket**
"""
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💡 Try an Example")

example_questions = [

    "What attendance percentage do I need for FAT?",

    "How late can I submit an assignment?",

    "When can I request revaluation?",

    "What are the library timings on Saturday?",

    "What is the WiFi password for the boys hostel?",
]


selected_question = st.selectbox(
    "Select a sample question",
    [""] + example_questions,
)


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_area(

    "🎓 Student Question",

    value=selected_question,

    placeholder=(
        "Example: What attendance percentage "
        "do I need for FAT?"
    ),

    height=100,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🤖 Ask Helpdesk",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Gemini is processing your question..."
        ):

            try:

                answer = run_agent(
                    question.strip(),
                    verbose=False,
                )

                st.subheader("💬 Helpdesk Response")

                st.success(answer)

            except Exception as error:

                st.error(
                    f"❌ Error while processing the question:\n\n"
                    f"{error}"
                )


# ============================================================
# SUPPORT TICKETS
# ============================================================

if TICKETS:

    st.divider()

    st.subheader("🎫 Support Tickets")

    for ticket in reversed(TICKETS):

        with st.expander(
            f"Ticket #{ticket['id']} — {ticket['created_at']}"
        ):

            st.write(
                "**Question:**"
            )

            st.write(
                ticket["question"]
            )

            st.write(
                "**Reason:**"
            )

            st.write(
                ticket["reason"]
            )

            st.info(
                f"Ticket #{ticket['id']} "
                "has been queued for the support team."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Grounded Student Helpdesk | "
    "Gemini • Embeddings • RAG • Agentic AI"
# )