import datetime
import json
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

# Gemini models
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.8-flash"

# Fallback generation model
FALLBACK_MODEL = "gemini-3.6-flash"

# Minimum similarity required to consider a policy relevant
MIN_SCORE = 0.40

# Maximum automatic tool calls
MAX_TOOL_CALLS = 6


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():
    """
    Create the Gemini API client using Streamlit Secrets.
    """

    try:
        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception as error:

        raise RuntimeError(
            "GEMINI_API_KEY was not found in Streamlit Secrets.\n\n"
            "Go to Streamlit Cloud → Settings → Secrets and add:\n\n"
            'GEMINI_API_KEY = "YOUR_API_KEY"'
        ) from error

    if not api_key or not api_key.strip():

        raise RuntimeError(
            "GEMINI_API_KEY is empty."
        )

    return genai.Client(
        api_key=api_key
    )


client = get_gemini_client()


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POLICY_FILE = BASE_DIR / "policies.json"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

if not POLICY_FILE.exists():

    raise FileNotFoundError(
        f"Could not find policies.json at:\n{POLICY_FILE}"
    )


with open(
    POLICY_FILE,
    "r",
    encoding="utf-8"
) as file:

    POLICIES = json.load(file)


# ============================================================
# SUPPORT TICKETS
# ============================================================

TICKETS = []


# ============================================================
# CHECK GEMINI
# ============================================================

def check_gemini():
    """
    Check whether the Gemini API key is configured.

    This does NOT send a Gemini generation request.
    Therefore, it does not consume a model request.
    """

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        if not api_key or not api_key.strip():

            raise RuntimeError(
                "GEMINI_API_KEY is empty."
            )

        print(
            "Gemini API key configured."
        )

        return True

    except Exception as error:

        raise RuntimeError(
            "Gemini API configuration failed.\n\n"
            "Make sure GEMINI_API_KEY is present "
            "in Streamlit Secrets.\n\n"
            f"Error: {error}"
        ) from error


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def embed_documents(docs):
    """
    Convert policy documents into embeddings
    using Gemini Embedding.
    """

    formatted_documents = []

    for document in docs:

        formatted = (
            f"title: {document['title']} | "
            f"text: {document['text']}"
        )

        formatted_documents.append(
            formatted
        )

    try:

        result = client.models.embed_content(

            model=EMBED_MODEL,

            contents=formatted_documents,

            config=types.EmbedContentConfig(

                task_type="RETRIEVAL_DOCUMENT",

                output_dimensionality=768,
            ),
        )

    except Exception as error:

        raise RuntimeError(
            "Gemini document embedding failed:\n"
            f"{error}"
        ) from error

    embeddings = np.asarray(
        [
            embedding.values
            for embedding in result.embeddings
        ],
        dtype=np.float32,
    )

    return embeddings


def embed_query(query):
    """
    Convert a student's question into an embedding.
    """

    formatted_query = (
        f"task: search result | "
        f"query: {query}"
    )

    try:

        result = client.models.embed_content(

            model=EMBED_MODEL,

            contents=formatted_query,

            config=types.EmbedContentConfig(

                task_type="RETRIEVAL_QUERY",

                output_dimensionality=768,
            ),
        )

    except Exception as error:

        raise RuntimeError(
            "Gemini query embedding failed:\n"
            f"{error}"
        ) from error

    embedding = np.asarray(
        result.embeddings[0].values,
        dtype=np.float32,
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_vector,
    document_matrix
):
    """
    Calculate cosine similarity between the query
    and every policy document.
    """

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:

        return np.zeros(
            document_matrix.shape[0],
            dtype=np.float32,
        )

    normalized_query = (
        query_vector / query_norm
    )

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
        keepdims=True,
    )

    document_norms = np.maximum(
        document_norms,
        1e-12,
    )

    normalized_documents = (
        document_matrix / document_norms
    )

    scores = (
        normalized_documents
        @ normalized_query
    )

    return scores


# ============================================================
# BUILD EMBEDDING INDEX
# ============================================================

POLICY_EMBEDDINGS = None


def build_embedding_index():
    """
    Create embeddings for the complete policy
    knowledge base.
    """

    global POLICY_EMBEDDINGS

    print(
        "\nCreating policy embeddings..."
    )

    POLICY_EMBEDDINGS = embed_documents(
        POLICIES
    )

    print(
        "Embedding matrix shape:",
        POLICY_EMBEDDINGS.shape
    )


# ============================================================
# POLICY SEARCH TOOL
# ============================================================

def search_policies(
    query: str
) -> str:
    """
    Search the university policy knowledge base.

    Uses Gemini embeddings and cosine similarity.

    Args:
        query: The student's question or topic.
    """

    if POLICY_EMBEDDINGS is None:

        return (
            "Policy knowledge base is not initialized."
        )

    query_vector = embed_query(
        query
    )

    scores = cosine_similarity(
        query_vector,
        POLICY_EMBEDDINGS,
    )

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_index]
    )

    print(
        f"\n[Retrieval] Best similarity score: "
        f"{best_score:.4f}"
    )

    if best_score < MIN_SCORE:

        return (
            "No confident match found in "
            "the policy knowledge base."
        )

    policy = POLICIES[
        best_index
    ]

    return (
        f"[{policy['title']}] "
        f"{policy['text']}"
    )


# ============================================================
# SUPPORT TICKET TOOL
# ============================================================

def create_support_ticket(
    question: str,
    reason: str
) -> str:
    """
    Create a support ticket when the knowledge base
    cannot confidently answer the student's question.

    Args:
        question: The student's original question.
        reason: Why human support is required.
    """

    ticket = {

        "id": len(TICKETS) + 1,

        "question": question,

        "reason": reason,

        "created_at": (
            datetime.datetime.now()
            .strftime("%Y-%m-%d %H:%M")
        ),
    }

    TICKETS.append(
        ticket
    )

    return (
        f"Ticket #{ticket['id']} "
        f"created and queued for the support team."
    )


# ============================================================
# TOOLS AVAILABLE TO GEMINI
# ============================================================

TOOLS = [
    search_policies,
    create_support_ticket,
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a grounded student helpdesk agent.

Your job is to answer student questions using
the available tools.

IMPORTANT RULES:

1. For every factual question about university,
   workshop, hostel, library, attendance,
   assignment, examination, or other policy,
   ALWAYS call search_policies first.

2. Never answer a university policy question
   using your own memory.

3. If search_policies returns a confident match,
   answer ONLY using the information returned
   by search_policies.

4. When answering from a policy, include the
   policy title in square brackets.

5. If search_policies returns:
   "No confident match found in the policy
   knowledge base."

   then call create_support_ticket.

6. When creating a ticket, use the student's
   original question and explain that the
   information is not available in the
   knowledge base.

7. NEVER invent university policies,
   passwords, rules, dates, fees, or procedures.

8. Be concise, clear, and helpful.

9. If the question is unrelated to university
   policy, answer normally if it can be answered
   safely without inventing university-specific
   information.
"""


# ============================================================
# AGENT
# ============================================================

def run_agent(
    user_message: str,
    max_turns: int = MAX_TOOL_CALLS,
    verbose: bool = True,
) -> str:
    """
    Run Gemini with automatic Python function calling.

    Primary:
        gemini-3.8-flash

    Fallback:
        gemini-2.5-flash

    Handles temporary 503 and 429 errors.
    """

    models_to_try = [
        GEN_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None

    for model_name in models_to_try:

        if verbose:

            print(
                f"\n[Agent] Using model: "
                f"{model_name}"
            )

        try:

            response = client.models.generate_content(

                model=model_name,

                contents=user_message,

                config=types.GenerateContentConfig(

                    system_instruction=SYSTEM_PROMPT,

                    tools=TOOLS,

                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=max_turns
                        )
                    ),
                ),
            )

            if response.text:

                if verbose:

                    print(
                        "\n[Agent] Final answer generated."
                    )

                return response.text

            return (
                "Gemini did not generate a response."
            )

        except Exception as error:

            last_error = error

            error_text = str(
                error
            ).upper()

            # ------------------------------------------------
            # TEMPORARY SERVICE ERROR
            # ------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                print(
                    f"\n[Agent] {model_name} "
                    "is temporarily unavailable."
                )

                continue

            # ------------------------------------------------
            # QUOTA ERROR
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                print(
                    f"\n[Agent] {model_name} "
                    "quota exceeded."
                )

                continue

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            return (
                "Error communicating with Gemini:\n"
                f"{error}"
            )

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    return (
        "Gemini is temporarily unavailable.\n\n"
        "Please try again in a few moments.\n\n"
        f"Technical details: {last_error}"
    )


# ============================================================
# ADD A NEW POLICY
# ============================================================

def add_policy(
    title: str,
    text: str
):
    """
    Add a new policy to the knowledge base
    and rebuild the embedding index.
    """

    POLICIES.append(
        {
            "title": title,
            "text": text,
        }
    )

    with open(
        POLICY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            POLICIES,
            file,
            indent=2,
            ensure_ascii=False,
        )

    build_embedding_index()

    print(
        f"\nNew policy added: {title}"
    )

    print(
        f"Knowledge base now contains "
        f"{len(POLICIES)} policies."
    )


# ============================================================
# DISPLAY TICKETS
# ============================================================

def display_tickets():

    print("\n")
    print("=" * 70)
    print("SUPPORT TICKETS")
    print("=" * 70)

    if not TICKETS:

        print(
            "No support tickets created."
        )

        return

    for ticket in TICKETS:

        print(
            f"\nTicket #{ticket['id']}"
        )

        print(
            f"Created: "
            f"{ticket['created_at']}"
        )

        print(
            f"Question: "
            f"{ticket['question']}"
        )

        print(
            f"Reason: "
            f"{ticket['reason']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("🎓 GROUNDED STUDENT HELPDESK AGENT")
    print("=" * 70)

    print(
        "\nTechnology:"
    )

    print(
        "  Generation Model :",
        GEN_MODEL,
    )

    print(
        "  Embedding Model  :",
        EMBED_MODEL,
    )

    print(
        "  Fallback Model   :",
        FALLBACK_MODEL,
    )

    print(
        "  Retrieval        : Cosine Similarity"
    )

    print(
        "  Architecture     : "
        "RAG + Agentic Tool Calling"
    )

    print(
        "  Knowledge Base   :",
        len(POLICIES),
        "policies",
    )

    # --------------------------------------------------------
    # STEP 1
    # Check Gemini
    # --------------------------------------------------------

    try:

        check_gemini()

    except Exception as error:

        print(
            "\n❌ Gemini setup error:"
        )

        print(error)

        raise SystemExit(1)

    # --------------------------------------------------------
    # STEP 2
    # Build embeddings
    # --------------------------------------------------------

    try:

        build_embedding_index()

    except Exception as error:

        print(
            "\n❌ Embedding error:"
        )

        print(error)

        raise SystemExit(1)

    # --------------------------------------------------------
    # STEP 3
    # Test questions
    # --------------------------------------------------------

    test_questions = [

        "What attendance percentage do I need for FAT?",

        "How late can I submit an assignment?",

        "When can I request revaluation?",

        "What are the library timings on Saturday?",

        "What is the WiFi password for the boys hostel?",
    ]

    # --------------------------------------------------------
    # STEP 4
    # Run tests
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNNING AGENT TESTS")
    print("=" * 70)

    for number, question in enumerate(
        test_questions,
        start=1,
    ):

        print("\n")
        print("-" * 70)

        print(
            f"TEST {number}"
        )

        print(
            f"Q: {question}"
        )

        answer = run_agent(
            question,
            verbose=True,
        )

        print(
            "\nA:",
            answer,
        )

    # --------------------------------------------------------
    # STEP 5
    # Display tickets
    # --------------------------------------------------------

    display_tickets()

    print("\n")
    print("=" * 70)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 70)
