import faiss
import pickle
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from .prompts import SYSTEM_PROMPT, USER_PROMPT

INDEX_PATH = "index/faiss_index.bin"
META_PATH = "index/meta.pkl"

class MLInterviewBot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.questions, self.answers = meta["questions"], meta["answers"]

        self.llm = pipeline("text-generation", model="google/flan-t5-base")

    def retrieve(self, query, top_k=3):
        q_emb = self.embedder.encode([query])
        D, I = self.index.search(q_emb, top_k)
        results = [(self.questions[i], self.answers[i]) for i in I[0]]
        return results

    def answer(self, query):
        docs = self.retrieve(query)
        context = "\n\n".join([f"Q: {q}\nA: {a}" for q,a in docs])
        prompt = SYSTEM_PROMPT + USER_PROMPT(query, context)
        output = self.llm(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        return output
