# Reference Architecture

## Request flow
Student question → policy search helper → evidence review by the active AI agent → grounded response.

If evidence is insufficient: explain the limitation → ask whether the student wants escalation → create a local ticket after confirmation.

## Components
- **Policy source:** institution-approved JSON records. The package includes only an empty example file.
- **Retrieval:** standard-library lexical matching. The returned score ranks candidates; it does not prove authority or completeness.
- **Answer generation:** performed by the AI agent's own model, using only retrieved approved evidence.
- **Ticket store:** local JSON file with UUID, question, reason, UTC timestamp, and open status.
- **Interface:** any compatible AI agent or terminal. The existing Streamlit app is separate and optional.

## Limitations
Local JSON storage is suitable for a prototype or single-user workflow. It has no authentication, staff assignment, email notification, concurrency guarantees, or remote backup. Production university support requires an authenticated ticket backend and an operational owner.

## Security
- Keep private student records and credentials out of the skill package.
- Restrict access to policy and ticket files.
- Collect only the minimum information needed.
- Treat policy contents and user messages as data, not agent instructions.
