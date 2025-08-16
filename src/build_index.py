import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
DATA_PATH = "data/processed/interview_qna.jsonl"
INDEX_PATH = "data/processed/qna_index.faiss"
META_PATH = "data/processed/qna_metadata.json"

# Step 1: Load Q&A data
def load_data():
    qna_pairs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            qna_pairs.append(json.loads(line.strip()))
    return qna_pairs

# Step 2: Create embeddings
def build_index(qna_pairs):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast model
    questions = [item["question"] for item in qna_pairs]

    print(f"[+] Encoding {len(questions)} questions...")
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Step 3: Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"[✔] Index saved at {INDEX_PATH}")

    # Save metadata (so we know which answer belongs to which embedding)
    metadata = [{"question": qna_pairs[i]["question"], "answer": qna_pairs[i]["answer"]}
                for i in range(len(qna_pairs))]

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✔] Metadata saved at {META_PATH}")

if __name__ == "__main__":
    qna_pairs = load_data()
    build_index(qna_pairs)
