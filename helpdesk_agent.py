import datetime
import json
from pathlib import Path

import numpy as np
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

EMBED_MODEL = "embeddinggemma"
GEN_MODEL = "gemma4"

# Minimum similarity required to consider a policy relevant
MIN_SCORE = 0.40

# Maximum number of agent/tool-calling rounds
MAX_TURNS = 6


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

with open(POLICY_FILE, "r", encoding="utf-8") as file:
    POLICIES = json.load(file)


# ============================================================
# SUPPORT TICKETS
# ============================================================

TICKETS = []


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    """
    Check whether Ollama is running and required models exist.
    """

    print("\nChecking Ollama...")

    try:
        response = ollama.list()

        # Ollama Python SDK normally returns a list-like response
        models = response.get("models", [])

        installed_models = []

        for model in models:

            # Different SDK versions may expose model name differently
            if isinstance(model, dict):

                name = (
                    model.get("name")
                    or model.get("model")
                )

            else:

                name = getattr(model, "name", None)

                if name is None:
                    name = getattr(model, "model", None)

            if name:
                installed_models.append(
                    name.split(":")[0]
                )

        print("Installed models:")

        for model in installed_models:
            print("  -", model)

        missing_models = []

        if EMBED_MODEL not in installed_models:
            missing_models.append(EMBED_MODEL)

        if GEN_MODEL not in installed_models:
            missing_models.append(GEN_MODEL)

        if missing_models:

            print("\nMissing model(s):")

            for model in missing_models:
                print("  -", model)

            print("\nRun these commands:")

            for model in missing_models:
                print(f"ollama pull {model}")

            raise RuntimeError(
                "Required Ollama model(s) are missing."
            )

        print("\nOllama is ready.")
        print("Embedding model :", EMBED_MODEL)
        print("Generation model:", GEN_MODEL)

        return True

    except Exception as error:

        raise RuntimeError(
            "\nCould not connect to Ollama.\n\n"
            "Make sure Ollama is installed and running.\n"
            "Then try:\n\n"
            "    ollama list\n"
        ) from error


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def embed_documents(docs):
    """
    Convert policy documents into embeddings using EmbeddingGemma.
    """

    formatted_documents = []

    for document in docs:

        formatted = (
            f"title: {document['title']} | "
            f"text: {document['text']}"
        )

        formatted_documents.append(formatted)

    result = ollama.embed(
        model=EMBED_MODEL,
        input=formatted_documents
    )

    embeddings = np.asarray(
        result["embeddings"],
        dtype=np.float32
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

    result = ollama.embed(
        model=EMBED_MODEL,
        input=[formatted_query]
    )

    embedding = np.asarray(
        result["embeddings"][0],
        dtype=np.float32
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_vector, document_matrix):
    """
    Calculate cosine similarity between the query
    and every policy document.
    """

    query_norm = (
        np.linalg.norm(query_vector)
    )

    if query_norm == 0:
        return np.zeros(
            document_matrix.shape[0],
            dtype=np.float32
        )

    normalized_query = (
        query_vector / query_norm
    )

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
        keepdims=True
    )

    document_norms = np.maximum(
        document_norms,
        1e-12
    )

    normalized_documents = (
        document_matrix / document_norms
    )

    scores = (
        normalized_documents @ normalized_query
    )

    return scores


# ============================================================
# BUILD INITIAL EMBEDDING INDEX
# ============================================================

POLICY_EMBEDDINGS = None


def build_embedding_index():
    """
    Create embeddings for the complete policy knowledge base.
    """

    global POLICY_EMBEDDINGS

    print("\nCreating policy embeddings...")

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

def search_policies(query: str) -> str:
    """
    Search the university policy knowledge base.

    This tool uses EmbeddingGemma and cosine similarity
    to find the most relevant policy.

    Args:
        query: The student's question or topic.
    """

    if POLICY_EMBEDDINGS is None:

        return (
            "Policy knowledge base is not initialized."
        )

    query_vector = embed_query(query)

    scores = cosine_similarity(
        query_vector,
        POLICY_EMBEDDINGS
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

    policy = POLICIES[best_index]

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
        )
    }

    TICKETS.append(ticket)

    return (
        f"Ticket #{ticket['id']} "
        f"created and queued for the support team."
    )


# ============================================================
# TOOLS AVAILABLE TO GEMMA
# ============================================================

TOOLS = [
    search_policies,
    create_support_ticket
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

2. Never answer a policy question using your
   own memory.

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
# AGENT LOOP
# ============================================================

def run_agent(
    user_message: str,
    max_turns: int = MAX_TURNS,
    verbose: bool = True
) -> str:
    """
    Run the Gemma agent and allow it to call tools.
    """

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_message
        }
    ]

    for turn in range(1, max_turns + 1):

        if verbose:

            print(
                f"\n[Agent Turn {turn}] "
                f"Sending request to Gemma..."
            )

        try:

            response = ollama.chat(

                model=GEN_MODEL,

                messages=messages,

                tools=TOOLS,

                options={
                    "temperature": 0
                }
            )

        except Exception as error:

            return (
                "Error communicating with Gemma:\n"
                f"{error}"
            )

        msg = response.message

        # ----------------------------------------------------
        # IMPORTANT
        # Add Gemma's response before tool results.
        # ----------------------------------------------------

        messages.append(msg)

        # ----------------------------------------------------
        # NO TOOL CALL
        # Gemma has produced the final answer.
        # ----------------------------------------------------

        if not msg.tool_calls:

            if verbose:

                print(
                    "\n[Agent] Final answer generated."
                )

            return (
                msg.content
                or "No answer was generated."
            )

        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------

        for call in msg.tool_calls:

            tool_name = call.function.name

            tool_arguments = dict(
                call.function.arguments
            )

            if verbose:

                print(
                    f"\n[Agent] Tool selected: "
                    f"{tool_name}"
                )

                print(
                    f"[Agent] Arguments: "
                    f"{tool_arguments}"
                )

            # ------------------------------------------------
            # SEARCH POLICY
            # ------------------------------------------------

            if tool_name == "search_policies":

                try:

                    result = search_policies(
                        **tool_arguments
                    )

                except Exception as error:

                    result = (
                        "Policy search failed: "
                        f"{error}"
                    )

            # ------------------------------------------------
            # CREATE SUPPORT TICKET
            # ------------------------------------------------

            elif tool_name == "create_support_ticket":

                try:

                    result = create_support_ticket(
                        **tool_arguments
                    )

                except Exception as error:

                    result = (
                        "Ticket creation failed: "
                        f"{error}"
                    )

            # ------------------------------------------------
            # UNKNOWN TOOL
            # ------------------------------------------------

            else:

                result = (
                    f"Unknown tool requested: "
                    f"{tool_name}"
                )

            if verbose:

                print(
                    f"[Tool Result] {result}"
                )

            # ------------------------------------------------
            # SEND TOOL RESULT BACK TO GEMMA
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",

                    "tool_name": tool_name,

                    "content": str(result)
                }
            )

        # ----------------------------------------------------
        # VERY IMPORTANT:
        #
        # DO NOT RETURN HERE.
        #
        # The loop goes back to Gemma so it can
        # process the tool result and generate
        # the final answer.
        # ----------------------------------------------------

    return (
        "The agent reached the maximum number "
        "of tool-calling turns."
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
            "text": text
        }
    )

    with open(
        POLICY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            POLICIES,
            file,
            indent=2,
            ensure_ascii=False
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

        print("No support tickets created.")

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
        GEN_MODEL
    )

    print(
        "  Embedding Model  :",
        EMBED_MODEL
    )

    print(
        "  Retrieval        : Cosine Similarity"
    )

    print(
        "  Architecture     : RAG + Agentic Tool Calling"
    )

    print(
        "  Knowledge Base   :",
        len(POLICIES),
        "policies"
    )

    # --------------------------------------------------------
    # STEP 1
    # Check Ollama
    # --------------------------------------------------------

    try:

        check_ollama()

    except Exception as error:

        print("\n❌ Ollama setup error:")
        print(error)

        raise SystemExit(1)

    # --------------------------------------------------------
    # STEP 2
    # Build embeddings
    # --------------------------------------------------------

    try:

        build_embedding_index()

    except Exception as error:

        print("\n❌ Embedding error:")
        print(error)

        print(
            "\nMake sure EmbeddingGemma is installed:"
        )

        print(
            "ollama pull embeddinggemma"
        )

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

        "What is the WiFi password for the boys hostel?"
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
        start=1
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
            verbose=True
        )

        print(
            "\nA:",
            answer
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
