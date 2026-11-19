"""Step 3 - Load documents and cut them into chunks."""
import os
from pypdf import PdfReader


def load_docs(folder="data/docs"):
    """Read every .txt, .md and .pdf file. Returns a list of (filename, text)."""
    docs = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.endswith((".txt", ".md")):
            docs.append((name, open(path, encoding="utf-8").read()))
        elif name.endswith(".pdf"):
            pages = PdfReader(path).pages
            docs.append((name, "\n".join(p.extract_text() or "" for p in pages)))
    return docs


def chunk(text, size=150, overlap=30):
    """Cut text into pieces of `size` words. Each piece repeats the last `overlap` words of the one before."""
    words = text.split()
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):   # reached the end, stop
            break
        start += size - overlap
    return pieces
def chunk_paragraphs(text, max_words=150):
    """One chunk per paragraph. Very long paragraphs are cut by words."""
    pieces = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para or (para.startswith("#") and len(para.split()) < 6):
            continue                      # skip empty lines and short headings
        if len(para.split()) > max_words:
            pieces.extend(chunk(para, max_words, 0))
        else:
            pieces.append(para)
    return pieces

def build_chunks(folder="data/docs", size=150, overlap=30, by="words"):
    """by='words' cuts by word count, by='paragraph' cuts at paragraph breaks."""
    result = []
    for name, text in load_docs(folder):
        pieces = chunk_paragraphs(text, size) if by == "paragraph" else chunk(text, size, overlap)
        for piece in pieces:
            result.append({"source": name, "text": piece})
    return result