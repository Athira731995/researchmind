from src.ingest import chunk, build_chunks

# 1. A tiny example so you can SEE the overlap
text = "one two three four five six seven eight nine ten"
for i, c in enumerate(chunk(text, size=4, overlap=1)):
    print(i, "->", c)

# 2. Your real document, cut into small chunks
print()
chunks = build_chunks(size=30, overlap=8)
print("Total chunks:", len(chunks))
for c in chunks[:3]:
    print(f"[{c['source']}] {c['text']}\n")