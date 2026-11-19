from src.ingest import build_chunks

chunks = build_chunks(by="paragraph")
print("Number of chunks:", len(chunks), "\n")
for i, c in enumerate(chunks):
    print(f"--- chunk {i} ({len(c['text'].split())} words) ---")
    print(c["text"][:120], "...\n")