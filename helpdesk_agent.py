import datetime
import json
from pathlib import Path

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

from helpdesk_core import JsonTicketStore, retrieve_policy


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
    """Create the Gemini API client using Streamlit Secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception as error:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in Streamlit Secrets.\n\n"
            "For Streamlit Cloud, add GEMINI_API_KEY in Secrets."
        ) from error

    if not api_key or not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is empty.")

    return genai.Client(api_key=api_key)


client = get_gemini_client()


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "policies.json"
TICKET_FILE = BASE_DIR / "tickets.json"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

if not POLICY_FILE.exists():
    raise FileNotFoundError(
        f"Could not find policies.json at:\n{POLICY_FILE}"
    )

with open(POLICY_FILE, "r", encoding="utf-8") as file:
    POLICIES = json.load(file)

if not isinstance(POLICIES, list):
    raise ValueError("policies.json must contain a JSON array.")


# ============================================================
# SUPPORT TICKETS
# ============================================================

TICKET_STORE = JsonTicketStore(TICKET_FILE)
TICKETS = TICKET_STORE.list_all()


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
            raise RuntimeError("GEMINI_API_KEY is empty.")

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
    """Convert policy documents into Gemini embeddings."""
    if not docs:
        return np.empty((0, 768), dtype=np.float32)

    formatted_documents = []

    for document in docs:
        formatted_documents.append(
            f"title: {document['title']} | text: {document['text']}"
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
        [embedding.values for embedding in result.embeddings],
        dtype=np.float32,
    )

    if len(embeddings) != len(docs):
        raise RuntimeError(
            "Gemini returned a different number of embeddings "
            "than the number of policy documents."
        )

    return embeddings


def embed_query(query):
    """Convert a student question into a Gemini embedding."""
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

def cosine_similarity(query_vector, document_matrix):
    """
    Calculate cosine similarity between the query
    and every policy document.
    """
    query_vector = np.asarray(query_vector, dtype=np.float32)
    document_matrix = np.asarray(document_matrix, dtype=np.float32)

    if document_matrix.size == 0:
        return np.asarray([], dtype=np.float32)

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return np.zeros(document_matrix.shape[0], dtype=np.float32)

    normalized_query = query_vector / query_norm

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
        keepdims=True,
    )
    document_norms = np.maximum(document_norms, 1e-12)

    normalized_documents = document_matrix / document_norms

    return normalized_documents @ normalized_query


# ============================================================
# BUILD EMBEDDING INDEX
# ============================================================

POLICY_EMBEDDINGS = None


def build_embedding_index():
    """Create embeddings for the complete policy knowledge base."""
    global POLICY_EMBEDDINGS

    POLICY_EMBEDDINGS = embed_documents(POLICIES)


# ============================================================
# POLICY SEARCH TOOL
# ============================================================

def search_policies(query: str) -> str:
    """
    Search the university policy knowledge base using
    the framework-independent retrieval function.
    """
    global LAST_RETRIEVAL

    if POLICY_EMBEDDINGS is None:
        LAST_RETRIEVAL = {
            "found": False,
            "title": None,
            "score": 0.0,
        }
        return "Policy knowledge base is not initialized."

    try:
        result = retrieve_policy(
            query=query,
            policies=POLICIES,
            document_embeddings=POLICY_EMBEDDINGS,
            embed_query=embed_query,
            threshold=MIN_SCORE,
        )

    except Exception as error:
        LAST_RETRIEVAL = {
            "found": False,
            "title": None,
            "score": 0.0,
        }
        return f"Policy search failed: {error}"

    policy = result["policy"]
    score = float(result["score"])

    if not result["found"] or policy is None:
        LAST_RETRIEVAL = {
            "found": False,
            "title": None,
            "score": score,
        }
        return (
            "No confident match found in "
            "the policy knowledge base."
        )

    LAST_RETRIEVAL = {
        "found": True,
        "title": policy["title"],
        "score": score,
    }

    return f"[{policy['title']}] {policy['text']}"


# ============================================================
# SUPPORT TICKET TOOL
# ============================================================

def create_support_ticket(question: str, reason: str) -> str:
    """
    Create and persist a support ticket when the
    knowledge base cannot answer a question.
    """
    global TICKETS

    try:
        ticket = TICKET_STORE.create(
            question=question,
            reason=reason,
        )

        # Refresh the existing list object so Streamlit imports see updates.
        TICKETS[:] = TICKET_STORE.list_all()

        return (
            f"Support ticket #{ticket['id']} "
            "created and saved successfully."
        )

    except Exception as error:
        return (
            "Support ticket creation failed. "
            f"Reason: {error}"
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

Your job is to answer student questions using the available tools.

IMPORTANT RULES:

1. For every factual question about university, workshop, hostel,
   library, attendance, assignment, examination, or other
   university policy, ALWAYS call search_policies first.

2. Never answer a university policy question using your own memory.

3. If search_policies returns a confident match, answer using ONLY
   the information returned by search_policies.

4. When answering from a policy, include the policy title in
   square brackets.

5. If search_policies returns:
   "No confident match found in the policy knowledge base."
   then call create_support_ticket.

6. When creating a ticket, use the student's original question.

7. NEVER invent university policies, passwords, rules, dates,
   fees, or procedures.

8. Be concise, clear, and helpful.

9. For unknown university information, explain that the
   information is unavailable. Only say a ticket was created
   if the ticket tool confirms successful creation.
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

    Primary model: gemini-3.8-flash
    Fallback model: gemini-3.6-flash
    """
    models_to_try = [
        GEN_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None

    for model_name in models_to_try:
        if verbose:
            print(f"\n[Agent] Using model: {model_name}")

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

            return "Gemini did not generate a response."

        except Exception as error:
            last_error = error
            error_text = str(error).upper()

            if "503" in error_text or "UNAVAILABLE" in error_text:
                if verbose:
                    print(
                        f"\n[Agent] {model_name} "
                        "is temporarily unavailable."
                    )
                continue

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                if verbose:
                    print(f"\n[Agent] {model_name} quota exceeded.")
                continue

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

def add_policy(title: str, text: str):
    """Add a new policy to the knowledge base."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Policy title must not be empty.")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Policy text must not be empty.")

    POLICIES.append({
        "title": title.strip(),
        "text": text.strip(),
    })

    with open(POLICY_FILE, "w", encoding="utf-8") as file:
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
    """Display all persisted support tickets."""
    tickets = TICKET_STORE.list_all()

    if not tickets:
        print("No support tickets created.")
        return

    for ticket in tickets:
        print(f"\nTicket #{ticket['id']}")
        print(f"Created: {ticket['created_at']}")
        print(f"Question: {ticket['question']}")
        print(f"Reason: {ticket['reason']}")
        print(f"Status: {ticket.get('status', 'open')}")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    print("\nGROUNDED STUDENT HELPDESK AGENT")

    print("\nGeneration Model:", GEN_MODEL)
    print("Embedding Model:", EMBED_MODEL)
    print("Fallback Model:", FALLBACK_MODEL)
    print("Policies:", len(POLICIES))

    check_gemini()
    build_embedding_index()

    question = input("\nAsk a question: ")

    answer = run_agent(question)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieval Information:")
    print(LAST_RETRIEVAL)

    display_tickets()
