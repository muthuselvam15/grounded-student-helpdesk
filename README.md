# 🎓 Grounded Student Helpdesk

An AI-powered university student helpdesk built using **Google Gemini, RAG, embeddings, cosine similarity, and agentic tool calling**.

The system answers student questions using a controlled university policy knowledge base and automatically creates a support ticket when the required information cannot be found.

---

## 🚀 Live Demo

🌐 **Streamlit App:**

https://grounded-student-ai.streamlit.app

---

## 📌 Problem Statement

Students frequently need quick answers about university policies such as:

- Attendance
- Examinations
- Revaluation
- Assignments
- Library rules
- Hostel information

A general-purpose AI model may generate information that is not actually part of the university's official policies.

This project addresses that problem by using **Retrieval-Augmented Generation (RAG)** to ground responses in a controlled university knowledge base.

---

## 💡 Solution

The **Grounded Student Helpdesk** combines RAG with an AI agent.

### When information is available:

```text
Student Question
        ↓
Gemini Query Embedding
        ↓
Policy Knowledge Base
        ↓
Cosine Similarity
        ↓
Relevant Policy
        ↓
Gemini Agent
        ↓
Grounded Answer
