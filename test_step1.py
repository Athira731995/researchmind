from src import llm

# 1. Basic call
resp = llm.chat([{"role": "user", "content": "Explain embeddings in one sentence."}])
print("BASIC:", llm.text_of(resp))
print("Tokens:", resp.prompt_eval_count, "in /", resp.eval_count, "out\n")

# 2. System prompt changes behaviour
resp = llm.chat([{"role": "user", "content": "Explain embeddings."}],
                system="You are a pirate. Answer in two sentences.")
print("SYSTEM PROMPT:", llm.text_of(resp), "\n")

# 3. Streaming
print("STREAM: ", end="")
for piece in llm.stream([{"role": "user", "content": "Count from 1 to 5 in words."}]):
    print(piece, end="", flush=True)
print("\n")

# 4. Structured output
schema = {"type": "object",
          "properties": {"name": {"type": "string"},
                         "skills": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "skills"]}
print("STRUCTURED:", llm.structured("Extract: Priya knows Python, SQL and Docker.", schema))

# 5. Embeddings
vecs = llm.embed(["cat", "kitten", "car"])
print("\nEMBED: vector length =", len(vecs[0]))