# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import spacy

# Load spaCy model once at startup
nlp = spacy.load("en_core_web_lg")

app = FastAPI()

# Health check route
@app.get("/")
def read_root():
    return {"hello": "world"}

# Request body for embedding
class TextRequest(BaseModel):
    text: str

# Endpoint: generate embeddings
@app.post("/embed")
def embed_text(request: TextRequest):
    doc = nlp(request.text)
    # Convert spaCy vector (numpy array) to Python list
    return {"vector": doc.vector.tolist()}

# Request body for similarity
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

# Endpoint: similarity between two texts
@app.post("/similarity")
def compute_similarity(request: SimilarityRequest):
    doc1 = nlp(request.text1)
    doc2 = nlp(request.text2)
    similarity = doc1.similarity(doc2)
    return {"similarity": similarity}