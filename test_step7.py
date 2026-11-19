from src import rag
from src.agent import run_agent

store = rag.build_store()

questions = [
    "Hello! Who are you?",                                            # no tool needed
    "What is 1234 * 5?",                                              # calculator only
    "What does self-attention do?",                                   # search only
    "Find how much LoRA cuts trainable parameters, then calculate how many of 5000 parameters would remain trainable.",
    ]

for q in questions:
    print("Q:", q)
    print("A:", run_agent(q, store))
    print()