from src import rag
from src.memory import Memory

store = rag.build_store()

print("WITHOUT memory")
print(rag.answer("How much does it reduce the trainable parameters?", store)["answer"], "\n")

print("WITH memory")
mem = Memory(window=2)
questions = [
    "What is LoRA?",
    "How much does it reduce the trainable parameters?",
    "What does RAG do?",
    "What problem does it reduce?",
]
for q in questions:
    out = rag.chat_turn(q, store, mem)
    print("Q:", q)
    print("  rewritten:", out["standalone"])
    print("  A:", out["answer"], "\n")

print("SUMMARY OF OLD TURNS:", mem.summary)