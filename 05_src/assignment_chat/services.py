import json
import os
import requests
from typing import List

import chromadb
from chromadb.config import Settings
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

from langchain.tools import tool



# Service 1: API Calls

@tool
def get_random_quote() -> str:
    """
    API-backed service using a stable public API.
    """
    url = "https://www.boredapi.com/api/activity"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        activity = data.get("activity", "Do something small and kind today.")
        kind = data.get("type", "general")
        participants = data.get("participants", 1)

        return f"Suggestion ({kind}, for {participants}): {activity}"

    except Exception:
        return "Suggestion: Step away from the screen for 2 minutes, stretch, then write one idea you’ve been postponing."



# Service 2: Semantic Query 

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection("mini_corpus")


def _ensure_seed_data() -> None:
    """
    Ensures there is at least a small dataset in Chroma so the demo works
    even if you haven't added your own dataset yet.
    """
    existing = _collection.count()
    if existing and existing > 0:
        return

    docs: List[str] = [
        "Affective polarization describes rising dislike and distrust between political groups, beyond policy disagreement.",
        "Moral panic is a process where a condition or group is framed as a threat to societal values and interests.",
        "Populism often constructs a frontier between 'the people' and 'the elite' and mobilizes grievances through that antagonism.",
        "Semantic search retrieves documents based on meaning similarity rather than exact keyword match.",
        "A vector database like Chroma stores embeddings to support similarity-based retrieval.",
    ]
    ids = [f"seed-{i}" for i in range(len(docs))]
    metas = [{"source": "seed"} for _ in docs]

    _collection.add(documents=docs, ids=ids, metadatas=metas)


@tool
def semantic_search(query: str) -> str:
    """
    Performs semantic search over a ChromaDB collection and returns top matches.
    """
    _ensure_seed_data()

    q = (query or "").strip()
    if not q:
        return "Please provide a non-empty search query."

    res = _collection.query(query_texts=[q], n_results=3)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if not docs:
        return "I couldn’t find anything relevant in the semantic index."

    lines = []
    for i, doc in enumerate(docs, start=1):
        src = (metas[i-1] or {}).get("source", "unknown")
        lines.append(f"{i}) {doc} [source: {src}]")

    return "Top semantic matches:\n" + "\n".join(lines)



# Service 3: Function Calling Tool 

@tool
def analyze_text(text: str) -> str:
    """
    Simple text analysis utility (word/sentence counts).
    """
    t = (text or "").strip()
    if not t:
        return "Provide some text after the tool call so I can analyze it."

    word_count = len(t.split())
    sentence_count = sum(t.count(x) for x in [".", "!", "?"])
    return f"Text stats: ~{word_count} words, ~{sentence_count} sentence-endings."