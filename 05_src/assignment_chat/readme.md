# Assignment Chat (InsightBot)

## Overview
InsightBot is a chat-based conversational AI system implemented using **Gradio** for the interface and **LangGraph/LangChain** for orchestration.  
The system provides three distinct services, enforces guardrails, and maintains short-term conversational memory.

---

## Services

### 1) API Service (`get_random_quote`)
- Uses an external public API (**Bored API / Activity**).
- Transforms the API response into natural language rather than returning raw JSON.

---

### 2) Semantic Query Service (`semantic_search`)
- Uses **ChromaDB PersistentClient** with on-disk persistence (`./chroma_store`).
- Supports semantic (meaning-based) retrieval.
- Includes a small seeded corpus for demonstration purposes.
- The corpus can be replaced with a custom dataset (≤ 40MB as per assignment constraints).

---

### 3) Function Calling Utility (`analyze_text`)
- Implemented as a callable tool available to the model.
- Provides lightweight text analysis (word and sentence counts).

---

## Memory
- Conversation history is preserved within the LangGraph state.
- A memory manager trims older messages to manage context window limitations.

---

## Guardrails
- Prevents access to or disclosure of system instructions.
- Prevents attempts to override or modify the system prompt.
- Enforces assignment-specific restricted topics:
  - Cats or dogs
  - Horoscopes or zodiac signs
  - Taylor Swift

---

## Running the Application
From the `assignment_chat` folder:

```bash
python app.py