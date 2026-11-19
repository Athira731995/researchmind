"""Step 4 - Vector store: meaning search, keyword search (BM25), hybrid, and query rewriting."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from src import llm


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None
        self.bm25 = None

    def add(self, chunks):
        """Index every chunk twice: as a vector (meaning) and for BM25 (keywords)."""
        self.chunks = chunks
        texts = ["search_document: " + c["text"] for c in chunks]
        vecs = np.array(llm.embed(texts), dtype="float32")
        self.vectors = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def meaning_rank(self, query):
        """Chunk indexes sorted best to worst by meaning."""
        q = np.array(llm.embed(["search_query: " + query])[0], dtype="float32")
        q = q / np.linalg.norm(q)
        scores = self.vectors @ q
        return list(np.argsort(scores)[::-1]), scores

    def keyword_rank(self, query):
        """Chunk indexes sorted best to worst by BM25 keyword score."""
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1])

    def search(self, query, k=3):
        """Meaning search only."""
        order, scores = self.meaning_rank(query)
        return [{**self.chunks[i], "score": float(scores[i])} for i in order[:k]]

    def hybrid(self, query, k=3):
        """Merge meaning and keyword rankings. A chunk near the top of either list scores well."""
        fused = {}
        for ranking in (self.meaning_rank(query)[0], self.keyword_rank(query)):
            for position, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (60 + position)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in best]


def rewrite(question):
    """Ask the model for a short textbook-style answer, to search with instead of the raw question."""
    prompt = ("Write a 2-sentence passage from a technical textbook that answers this question. "
              "Use precise technical terms.\nQuestion: " + question)
    return llm.text_of(llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=100))