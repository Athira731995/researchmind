"""Step 11 - API: let other programs talk to ResearchMind."""
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src import rag, agent
from src.memory import Memory

app = FastAPI(title="ResearchMind")
store = rag.build_store()          # built once, when the server starts
sessions = {}                      # one Memory per conversation


class Query(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    memory = sessions.setdefault(q.session_id, Memory())
    start = time.time()
    out = rag.chat_turn(q.question, store, memory)
    out["latency_seconds"] = round(time.time() - start, 1)
    return out


@app.post("/ask/stream")
def ask_stream(q: Query):
    return StreamingResponse(rag.answer_stream(q.question, store), media_type="text/plain")


@app.post("/agent")
def run_agent(q: Query):
    return {"answer": agent.run_agent(q.question, store, verbose=False)}