# Grounded Student Helpdesk — Agent Skill

A reusable instruction skill for building and maintaining a university helpdesk grounded in an approved policy knowledge base.

## What this skill helps an agent do

- Design a RAG-based university helpdesk.
- Structure and retrieve policy records.
- Generate answers grounded in retrieved policy text.
- Handle unanswered questions without hallucinating institutional rules.
- Design a support-ticket workflow with explicit success/failure handling.
- Review security, privacy, testing, and deployment considerations.

## Package contents

- `SKILL.md` — agent instructions and acceptance criteria.
- `references/architecture.md` — reference architecture and implementation caveats.
- `references/policy-format.md` — suggested policy record format.

## Important distinction

This package contains agent instructions and reference documentation. It is not the hosted Streamlit application, does not include a running Python service, and does not itself provide a persistent ticket backend or API credentials.

The original application remains in the repository root. Use its source as an implementation reference, and verify current dependencies and model availability before running it.

## Safety and data handling

Use only institution-approved policy documents. Do not commit secrets or real student records. Configure a persistent, access-controlled ticket backend before using the system for operational support.
