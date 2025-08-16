from fastapi import FastAPI
from pydantic import BaseModel
from .rag_pipeline import MLInterviewBot

app = FastAPI()
bot = MLInterviewBot()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    answer = bot.answer(q.question)
    return {"answer": answer}
