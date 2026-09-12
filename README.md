# 🎓 Grounded Student Helpdesk Agent

A capstone project built with **Ollama, Gemma 4, EmbeddingGemma, NumPy, RAG, and agentic tool calling**.

## What it does

The agent answers student policy questions using a local knowledge base.

- **Known question →** semantic retrieval → grounded answer → policy citation
- **Unknown question →** retrieval fails confidence check → support ticket is created
- **No hallucinated policy answer**

## Architecture

Student Question
→ Gemma 4 Agent
→ `search_policies()` tool
→ EmbeddingGemma
→ Cosine Similarity
→ Policy Knowledge Base
→ Grounded Answer

If confidence is below the threshold:

Gemma 4
→ `create_support_ticket()`
→ Human Support Queue

## Requirements

- Python 3.8+
- Ollama installed and running
- `gemma4`
- `embeddinggemma`

Ollama must be running locally on port 11434.

## Windows setup

1. Install Ollama from the official Ollama website.
2. Open PowerShell.
3. Run:

```powershell
ollama pull gemma4
ollama pull embeddinggemma
```

4. Create and activate a Python environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

5. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Test the CLI:

```powershell
python helpdesk_agent.py
```

7. Launch the UI:

```powershell
python -m streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Jupyter / Google Colab note

The original workshop notebook uses `!pip install`. That installs the Python SDK, but the SDK still needs an Ollama server. The easiest and most reliable setup is to run Ollama locally on your computer and execute this project locally.

If you use Colab, you need an Ollama server reachable by the notebook; simply running `pip install ollama` is not enough.

## Why the workshop code can fail

The final workshop cell calls:

```python
run_agent(your_question, tools=tools, system=helpdesk_system)
```

but `tools` must first be defined:

```python
tools = [search_policies, create_support_ticket]
```

The workshop handout defines this in its Step 7. Also, the model files must already be available to Ollama.

## Demo questions

Known:

- What attendance percentage do I need for FAT?
- How late can I submit an assignment?
- When can I request revaluation?
- What time does the library close on Saturday?

Unknown:

- What is the WiFi password for the boys hostel?
- Who is my faculty advisor?

Unknown questions should be escalated rather than answered from model memory.

## Technologies

- Python
- Ollama
- Gemma 4
- EmbeddingGemma
- NumPy
- Retrieval-Augmented Generation
- Cosine similarity
- Agentic tool calling
- Streamlit

## Extension ideas

1. Add PDF/document ingestion.

