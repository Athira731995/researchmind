from src.ingest import build_chunks
from src.store import VectorStore


def keyword_search(chunks, query, k=1):
    """Old-style search: count how many words the query shares with each chunk."""
    q = set(query.lower().split())
    scored = [(len(q & set(c["text"].lower().split())), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

chunks = build_chunks(by="paragraph")   
store = VectorStore()
store.add(chunks)
print("Indexed", len(chunks), "chunks\n")

queries = [
    "How do I stop the AI from making things up?",
    "How can a model be adapted cheaply?",
    "How does a model know which words matter?",
]

for q in queries:
    print("QUESTION:", q)
    best = store.search(q, k=1)[0]
    print(f"  MEANING (score {best['score']:.2f}): {best['text'][:200]}")
    score, kc = keyword_search(chunks, q)[0]
    print(f"  KEYWORD (shared words {score}): {kc['text'][:200]}\n")