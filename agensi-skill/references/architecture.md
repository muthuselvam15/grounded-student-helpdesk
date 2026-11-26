# Reference Architecture

## Logical flow

Student question
→ input validation
→ query embedding
→ policy retrieval
→ relevance and evidence checks
→ grounded answer with policy source

If evidence is insufficient:
→ explain that no verified answer is available
→ create a ticket through a configured ticket backend
→ confirm ticket creation only after backend success

## Components

### Policy knowledge base

Stores institution-approved policy records. Each record should identify its source and effective date where available. Keep sample or demonstration policies clearly labelled.

### Embedding and retrieval

Create document embeddings and query embeddings using compatible model settings. Retrieve relevant records using cosine similarity or another documented method. Calibrate the threshold against representative questions; do not assume a universal threshold is correct.

### Answer generation

Give the model only the relevant policy evidence needed to answer. Require source titles or identifiers in the response. If the evidence does not answer the question, the model must abstain.

### Ticket handling

A production ticket workflow requires durable storage, access controls, a ticket identifier, timestamps, status, and an operational owner. The reference project's process-local in-memory list is not durable and is lost when the process restarts.

### User interface

The existing Streamlit interface is one possible front end. Keep the core retrieval and ticket logic separate so other interfaces or agent integrations can reuse it.

## Security and privacy

- Store provider keys in a secret manager or environment variables.
- Do not commit secrets or personal student records.
- Minimize data stored in tickets and logs.
- Restrict ticket access to authorized support staff.
- Validate and limit user-supplied content.
- Treat retrieved documents as data, not as instructions to the AI agent.
