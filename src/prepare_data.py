import os
import json
import re
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "interview_qna.jsonl")

def parse_txt(file_path):
    """Parse numbered Q&A from TXT file"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    qna_list = []
    question = None
    answer_lines = []

    question_pattern = re.compile(r"^\d+\.\s*(.*)\?$")  # matches "1. Question?"

    for line in lines:
        match = question_pattern.match(line)
        if match:
            # Save previous Q&A
            if question and answer_lines:
                qna_list.append({
                    "question": question,
                    "answer": " ".join(answer_lines)
                })
            question = match.group(1).strip()
            answer_lines = []
        else:
            answer_lines.append(line)

    # Save last Q&A
    if question and answer_lines:
        qna_list.append({
            "question": question,
            "answer": " ".join(answer_lines)
        })

    return qna_list

def parse_csv(file_path):
    """Parse CSV with 'question','answer' columns"""
    df = pd.read_csv(file_path)
    qna_list = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a:
            qna_list.append({"question": q, "answer": a})
    return qna_list

def prepare_data():
    qna_total = []

    for fname in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, fname)
        if fname.endswith(".txt"):
            print(f"[+] Parsing TXT: {fname}")
            qna_total.extend(parse_txt(path))
        elif fname.endswith(".csv"):
            print(f"[+] Parsing CSV: {fname}")
            qna_total.extend(parse_csv(path))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in qna_total:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[✔] Processed {len(qna_total)} Q&A pairs into {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
