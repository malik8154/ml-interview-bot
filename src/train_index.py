import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "interview_qna.jsonl")
INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "qna.index")
META_PATH = os.path.join(BASE_DIR, "data", "index", "metadata.json")

# Ensure output directory exists
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# Load processed data
print("[+] Loading processed Q&A data...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Load embedding model
print("[+] Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all questions
print(f"[+] Generating embeddings for {len(questions)} questions...")
embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index
faiss.write_index(index, INDEX_PATH)

# Save metadata (questions + answers)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump({"questions": questions, "answers": answers}, f, indent=2)

print(f"[✔] Index saved at {INDEX_PATH}")
print(f"[✔] Metadata saved at {META_PATH}")
