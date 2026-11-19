from src.ingest import build_chunks
from src.store import VectorStore, rewrite

chunks = build_chunks(by="paragraph")
store = VectorStore()
store.add(chunks)


def rank_of(results, marker="self-attention"):
    for position, r in enumerate(results, start=1):
        if marker in r["text"]:
            return position
    return "missing"


question = "How does a model know which words matter?"
fake_answer = rewrite(question)
print("Rewritten query:", fake_answer, "\n")

n = len(chunks)
print("Rank of the self-attention chunk (1 = best):")
print("  Meaning only:          ", rank_of(store.search(question, k=n)))
print("  Hybrid:                ", rank_of(store.hybrid(question, k=n)))
print("  Meaning + rewrite:     ", rank_of(store.search(fake_answer, k=n)))
print("  Hybrid + rewrite:      ", rank_of(store.hybrid(fake_answer, k=n)))