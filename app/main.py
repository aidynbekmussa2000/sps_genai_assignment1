# app/main.py
from contextlib import asynccontextmanager
from typing import List
import numpy as np
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlp = spacy.load("en_core_web_lg")  # load once at startup
    yield

app = FastAPI(title="Embeddings API", lifespan=lifespan)

class TextIn(BaseModel):
    text: str

class PairIn(BaseModel):
    a: str
    b: str

def to_vec(text: str, nlp) -> np.ndarray:
    doc = nlp(text)
    if doc.vector_norm == 0:
        raise HTTPException(status_code=400, detail="Zero vector (no word vectors).")
    return doc.vector

@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post("/embed")
def embed(payload: TextIn):
    v = to_vec(payload.text, app.state.nlp)
    return {"dim": int(v.shape[0]), "vector": v.tolist()}

@app.post("/similarity")
def similarity(payload: PairIn):
    v1 = to_vec(payload.a, app.state.nlp)
    v2 = to_vec(payload.b, app.state.nlp)
    sim = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return {"similarity": sim}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)