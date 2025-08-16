# ML Interview Q\&A Bot

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange?logo=streamlit) ![FAISS](https://img.shields.io/badge/FAISS-vector%20search-green)

A Python-based **Machine Learning Interview Q\&A Bot** that answers ML interview questions using **semantic search**. It leverages **FAISS** for fast vector similarity search, **sentence-transformers** for embeddings, and **Streamlit** for an interactive web interface.

---

## Features

* ✅ Parse Q\&A datasets from TXT & CSV files
* ✅ Generate embeddings with `all-MiniLM-L6-v2`
* ✅ Build and search FAISS vector index
* ✅ Interactive Streamlit web interface
* ✅ Easily extendable with custom Q\&A datasets

---

## Folder Structure

```
ml-interview-bot/
│
├─ app/                   # Streamlit UI
│   └─ ui.py
├─ data/
│   ├─ raw/               # Original TXT & CSV files
│   └─ processed/         # JSONL after prepare_data.py
├─ data/index/            # FAISS index + metadata (generated locally, not uploaded)
├─ src/                   # Python scripts
│   ├─ prepare_data.py
│   ├─ train_index.py
│   └─ query_index.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/malik8154/ml-interview-bot.git
cd ml-interview-bot
```

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Prepare the data**

```bash
python src/prepare_data.py
```

5. **Build the FAISS index**

```bash
python src/train_index.py
```

6. **Run the Streamlit UI**

```bash
streamlit run app/ui.py
```

---

## Usage

* Enter any ML interview question in the text box.
* Get **top 3 most relevant answers** retrieved from your dataset.
* Add your own questions in `data/raw` and rerun `prepare_data.py` + `train_index.py` to update the bot.

---

## Notes

* The FAISS index (`data/index/qna.index`) and metadata are **not included** in the repository due to size. They are generated locally.
* Works with Python 3.11+.
* Keep your virtual environment and cache directories out of GitHub (`.gitignore` handles this).

---

## License

MIT License © 2025 Malik  M Shahmeer Rashid

---

This README is **professional, detailed, and beginner-friendly**, so anyone can clone your repo and run the bot.

If you want, I can also **suggest GitHub topics/tags and a short repo description** so it’s easier to discover. Do you want me to do that?
