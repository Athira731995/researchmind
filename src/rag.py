"""Step 6 - RAG: retrieve the right chunks, then answer from them with citations."""
from src import llm, prompts, guardrails
from src.ingest import build_chunks
from src.store import VectorStore, rewrite


def build_store(by="paragraph"):
    """Load documents, chunk them, and index them for search."""
    store = VectorStore()
    store.add(build_chunks(by=by))
    return store


def retrieve(question, store, k=3):
    """Find the k best chunks, then strip out any attack sentences."""
    chunks = store.hybrid(question + " " + rewrite(question), k=k)
    return guardrails.clean_context(chunks)


def answer(question, store, k=3, cot=False):
    ok, q = guardrails.check_input(question)       # checkpoint 1
    if not ok:
        return {"answer": q, "sources": []}

    chunks = retrieve(q, store, k)                  # search + checkpoint 2
    if not chunks:
        return {"answer": "I don't have enough information.", "sources": []}

    context = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    user = prompts.build_user_prompt(context, q, cot)
    msgs = prompts.FEW_SHOT + [{"role": "user", "content": user}]
    raw = llm.text_of(llm.chat(msgs, system=prompts.RAG_SYSTEM, temperature=0))
    text = prompts.extract_answer(raw)

    if not guardrails.check_output(text, len(chunks)):    # checkpoint 3
        text = "I don't have enough information."          # uncited answers are discarded

    return {"answer": text,
            "sources": [{"id": i + 1, "file": c["source"], "text": c["text"][:100]}
                        for i, c in enumerate(chunks)]}
def answer_stream(question, store, k=3):
    """Same as answer(), but yields the reply piece by piece."""
    ok, q = guardrails.check_input(question)
    if not ok:
        yield q
        return
    chunks = retrieve(q, store, k)
    if not chunks:
        yield "I don't have enough information."
        return
    context = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    user = prompts.build_user_prompt(context, q)
    msgs = prompts.FEW_SHOT + [{"role": "user", "content": user}]
    yield from llm.stream(msgs, system=prompts.RAG_SYSTEM)
def chat_turn(question, store, memory):
    """One turn of a conversation: check, rewrite using memory, answer, then remember."""
    ok, q = guardrails.check_input(question)          # check the ORIGINAL question first
    if not ok:
        return {"answer": q, "sources": [], "standalone": q}
    standalone = memory.standalone(q)
    out = answer(standalone, store)
    if "don't have enough information" not in out["answer"].lower():
        memory.add("user", q)                      # only remember turns that worked
        memory.add("assistant", out["answer"])
    out["standalone"] = standalone
    return out