import os, time, json
import ollama
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL", "llama3.2")
EMBED_MODEL = "nomic-embed-text"


def chat(messages, system="", max_tokens=1000, tools=None, fmt=None, temperature=0.2):
    """One call to the local model, retried with exponential backoff."""
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    for attempt in range(3):
        try:
            return ollama.chat(model=MODEL, messages=msgs, tools=tools, format=fmt,
                               options={"num_predict": max_tokens, "temperature": temperature})
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def text_of(resp):
    return resp.message.content or ""


def stream(messages, system="", max_tokens=1000):
    """Yields text pieces as the model generates them."""
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    for chunk in ollama.chat(model=MODEL, messages=msgs, stream=True,
                             options={"num_predict": max_tokens}):
        yield chunk.message.content


def structured(prompt, schema):
    """Ollama enforces the JSON schema while generating, so the output always parses."""
    resp = chat([{"role": "user", "content": prompt}], fmt=schema)
    return json.loads(text_of(resp))


def embed(texts):
    """Turn a list of strings into vectors."""
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings