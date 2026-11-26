"""Framework-independent core utilities for Grounded Student Helpdesk.

This module contains deterministic retrieval helpers and durable ticket storage.
It deliberately does not import Streamlit or create an AI provider client.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


def cosine_similarity(query_vector: Sequence[float],
                      document_matrix: Sequence[Sequence[float]]) -> list[float]:
    """Return cosine similarity between one query vector and each document vector."""
    query = [float(value) for value in query_vector]
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        return [0.0 for _ in document_matrix]

    scores: list[float] = []
    for document in document_matrix:
        vector = [float(value) for value in document]
        if len(vector) != len(query):
            raise ValueError("Query and document vectors must have equal dimensions.")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            scores.append(0.0)
        else:
            scores.append(sum(a * b for a, b in zip(query, vector)) / (query_norm * norm))
    return scores


def validate_policies(policies: Iterable[dict]) -> list[dict]:
    """Validate and normalize the minimal title/text policy schema."""
    normalized = []
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise ValueError(f"Policy at index {index} must be an object.")
        title = policy.get("title")
        text = policy.get("text")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Policy at index {index} needs a non-empty title.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Policy at index {index} needs non-empty text.")
        normalized.append({**policy, "title": title.strip(), "text": text.strip()})
    return normalized


def retrieve_policy(query: str,
                    policies: Sequence[dict],
                    document_embeddings: Sequence[Sequence[float]],
                    embed_query: Callable[[str], Sequence[float]],
                    threshold: float = 0.40) -> dict:
    """Retrieve the best policy; return found=False when evidence is below threshold."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    checked_policies = validate_policies(policies)
    if len(checked_policies) != len(document_embeddings):
        raise ValueError("Policy and embedding counts must match.")
    if not checked_policies:
        return {"found": False, "policy": None, "score": 0.0}

    query_vector = embed_query(query)
    scores = cosine_similarity(query_vector, document_embeddings)
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best_index]
    found = best_score >= threshold
    return {
        "found": found,
        "policy": checked_policies[best_index] if found else None,
        "score": best_score,
    }


class JsonTicketStore:
    """Small local JSON ticket store; use a database for multi-worker production."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create(self, question: str, reason: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Ticket question must not be empty.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Ticket reason must not be empty.")

        tickets = self.list_all()
        ticket = {
            "id": str(uuid.uuid4()),
            "question": question.strip(),
            "reason": reason.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }
        tickets.append(ticket)
        self._write(tickets)
        return ticket

    def list_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read ticket store: {error}") from error
        if not isinstance(data, list):
            raise RuntimeError("Ticket store must contain a JSON array.")
        return data

    def _write(self, tickets: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                delete=False, prefix=f".{self.path.name}.", suffix=".tmp"
            ) as handle:
                json.dump(tickets, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                temp_name = handle.name
            os.replace(temp_name, self.path)
        except OSError as error:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
            raise RuntimeError(f"Could not save support ticket: {error}") from error
