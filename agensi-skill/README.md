# Grounded Student Helpdesk — Agensi Skill

A portable agent skill for answering university questions from institution-approved policy files and recording unanswered questions in a local JSON ticket store.

## Contents
- `SKILL.md`: agent workflow and safety rules.
- `scripts/policy_search.py`: standard-library policy search helper.
- `scripts/ticket_store.py`: persistent local JSON ticket helper.
- `references/architecture.md`: architecture and limitations.
- `references/policy-format.md`: policy JSON format.
- `data/policies.example.json`: intentionally empty starter file.

## Requirements
Python 3.9+. The bundled scripts use only the Python standard library. No Gemini API key, Streamlit server, or third-party package is required. The AI agent uses its own available model.

## Quick start
1. Copy `data/policies.example.json` to `data/policies.json`.
2. Add institution-approved policy records.
3. Search:
   `python scripts/policy_search.py --query "attendance requirement" --policies data/policies.json`
4. After the user confirms escalation, create a ticket:
   `python scripts/ticket_store.py create --question "Unknown deadline?" --reason "No approved policy match."`
5. List tickets:
   `python scripts/ticket_store.py list`

## Important
Tickets are stored locally. They are not sent to university staff or an official ticketing system. The example policy file is empty; add only current, approved institutional content. Never package private student data or API credentials.
