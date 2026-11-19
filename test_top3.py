from src.ingest import build_chunks
from src.store import VectorStore

chunks = build_chunks(by="paragraph")
store = VectorStore()
store.add(chunks)

question = "How does a model know which words matter?"
print("QUESTION:", question, "\n")
for rank, r in enumerate(store.search(question, k=3), start=1):
    print(f"#{rank} (score {r['score']:.2f}): {r['text'][:80]}...")