import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="ML Interview Q&A Bot", page_icon="🤖")
st.title("🤖 ML Interview Q&A Bot")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "index", "qna.index")
META_PATH = os.path.join(BASE_DIR, "data", "index", "metadata.json")

# Load embedding model
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
@st.cache_resource(show_spinner=True)
def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta["questions"], meta["answers"]

model = load_model()
index, questions, answers = load_index()

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

# Streamlit input
user_input = st.text_area("Ask your ML interview question here:")

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Searching for answers..."):
            results = query(user_input)
            st.success("Top Answers Found:")

            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**A:** {r['answer']}")
                st.markdown(f"_Score: {r['score']:.4f}_")
                st.markdown("---")
