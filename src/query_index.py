import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "qna.index")
META_PATH = os.path.join(BASE_DIR, "data", "index", "metadata.json")

# Load embedding model
print("[+] Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
print(f"[+] Loading FAISS index from {INDEX_PATH}...")
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

questions = meta["questions"]
answers = meta["answers"]

# Function to query
def query(question, top_k=3):
    embedding = model.encode([question], convert_to_numpy=True)
    D, I = index.search(embedding, top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(dist)
        })
    return results

# Interactive loop
if __name__ == "__main__":
    while True:
        q = input("\n❓ Enter your interview question (or 'exit' to quit): ")
        if q.lower() in ["exit", "quit"]:
            break

        results = query(q)
        print("\n🔎 Top Answers:")
        for r in results:
            print(f"\nQ: {r['question']}\nA: {r['answer']}\n(score: {r['score']:.4f})")
