# Policy Knowledge-Base Format

## Suggested record

Each policy record should include:

- `id`: stable unique identifier
- `title`: short human-readable policy name
- `text`: approved policy wording
- `source`: official document or URL identifier
- `effective_from`: optional effective date
- `reviewed_at`: optional last verification date
- `owner`: optional responsible department

## Example schema

```json
[
  {
    "id": "attendance-001",
    "title": "Attendance Policy",
    "text": "Insert the exact approved policy wording here.",
    "source": "Official academic regulations, section 4.2",
    "effective_from": "2026-07-01",
    "reviewed_at": "2026-07-01",
    "owner": "Academic Office"
  }
]
```

The wording above is a placeholder, not an actual university rule.

## Quality rules

1. Use official, current source material.
2. Keep one policy topic per record where practical.
3. Preserve exceptions, conditions, dates, and scope.
4. Record conflicting versions and resolve them with the policy owner before deployment.
5. Rebuild embeddings whenever approved policy text changes.
6. Do not use the example policies in the original demo JSON as authoritative rules without institutional verification.
