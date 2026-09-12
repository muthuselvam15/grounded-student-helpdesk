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

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.6-flash"

MIN_SCORE = 0.40
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
            "For Streamlit Cloud, add GEMINI_API_KEY in Secrets."
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
# RAG RETRIEVAL STATUS
# ============================================================

LAST_RETRIEVAL = {
    "found": False,
    "title": None,
    "score": 0.0,
}


# ============================================================
# CHECK GEMINI
# ============================================================

def check_gemini():
    """
    Check whether the Gemini API key is configured.

    This does not make a generation request.
    """

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        if not api_key or not api_key.strip():
            raise RuntimeError(
                "GEMINI_API_KEY is empty."
            )

        return True

    except Exception as error:

        raise RuntimeError(
            "Gemini API configuration failed.\n\n"
            f"Error: {error}"
        ) from error


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def embed_documents(docs):
    """
    Convert policy documents into Gemini embeddings.
    """

    formatted_documents = []

    for document in docs:

        formatted_documents.append(
            f"title: {document['title']} | "
            f"text: {document['text']}"
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
    Convert a student question into a Gemini embedding.
    """

    try:

        result = client.models.embed_content(

            model=EMBED_MODEL,

            contents=query,

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

    return np.asarray(
        result.embeddings[0].values,
        dtype=np.float32,
    )


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

    return (
        normalized_documents
        @ normalized_query
    )


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

    POLICY_EMBEDDINGS = embed_documents(
        POLICIES
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
    """

    global LAST_RETRIEVAL

    if POLICY_EMBEDDINGS is None:

        LAST_RETRIEVAL = {
            "found": False,
            "title": None,
            "score": 0.0,
        }

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

    # --------------------------------------------------------
    # NO CONFIDENT MATCH
    # --------------------------------------------------------

    if best_score < MIN_SCORE:

        LAST_RETRIEVAL = {
            "found": False,
            "title": None,
            "score": best_score,
        }

        return (
            "No confident match found in "
            "the policy knowledge base."
        )

    # --------------------------------------------------------
    # CONFIDENT MATCH
    # --------------------------------------------------------

    policy = POLICIES[
        best_index
    ]

    LAST_RETRIEVAL = {
        "found": True,
        "title": policy["title"],
        "score": best_score,
    }

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
    cannot answer the question.
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
# TOOLS
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
   assignment, examination, or other university
   policy, ALWAYS call search_policies first.

2. Never answer a university policy question
   using your own memory.

3. If search_policies returns a confident match,
   answer using ONLY the information returned
   by search_policies.

4. When answering from a policy, include the
   policy title in square brackets.

5. If search_policies returns:
   "No confident match found in the policy
   knowledge base."

   then call create_support_ticket.

6. When creating a ticket, use the student's
   original question.

7. NEVER invent university policies,
   passwords, rules, dates, fees, or procedures.

8. Be concise, clear, and helpful.

9. For unknown university information, explain
   that the information is unavailable and that
   a support ticket has been created.
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

    Primary model:
        gemini-3.8-flash

    Fallback model:
        gemini-3.6-flash
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

    return (
        "Gemini is temporarily unavailable.\n\n"
        "Please try again in a few moments.\n\n"
        f"Technical details: {last_error}"
    )


# ============================================================
# ADD POLICY
# ============================================================

def add_policy(
    title: str,
    text: str
):
    """
    Add a new policy to the knowledge base.
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


# ============================================================
# DISPLAY TICKETS
# ============================================================

def display_tickets():

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

    print(
        "\n🎓 GROUNDED STUDENT HELPDESK AGENT"
    )

    print(
        "\nGeneration Model:",
        GEN_MODEL,
    )

    print(
        "Embedding Model:",
        EMBED_MODEL,
    )

    print(
        "Fallback Model:",
        FALLBACK_MODEL,
    )

    print(
        "Policies:",
        len(POLICIES),
    )

    check_gemini()

    build_embedding_index()

    question = input(
        "\nAsk a question: "
    )

    answer = run_agent(
        question
    )

    print(
        "\nAnswer:\n"
    )

    print(
        answer
    )

    print(
        "\nRetrieval Information:"
    )

    print(
        LAST_RETRIEVAL
    )

    display_tickets()
