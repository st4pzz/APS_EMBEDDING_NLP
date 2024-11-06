from fastapi import FastAPI, Query, Response
import uvicorn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

app = FastAPI()


df = pd.read_csv('dataset/filtered_lyrics.csv')


embeddings = np.load('embeddings/lyrics_embeddings.npy')


sbert_model = SentenceTransformer('all-mpnet-base-v2')


def clean_text(text):
    
    text = text.lower()
 
    text = re.sub(r'[^a-z0-9\s]', '', text)
   
    text = re.sub(r'\s+', ' ', text)
   
    text = text.strip()
    return text


def get_query_embedding(query):
   
    query = clean_text(query)
    query_embedding = sbert_model.encode(query, convert_to_numpy=True)
    return query_embedding


def search(query, top_k=10):
    query_embedding = get_query_embedding(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    sorted_indices = similarities.argsort()[::-1]
    results = []
    titles = set()
    for idx in sorted_indices:
        title = df.iloc[idx]['Song Name']
        if title not in titles:
            results.append({
                'Song Name': title,
                'Lyrics': df.iloc[idx]['Lyrics'],
                'Similarity': float(similarities[idx])
            })
            titles.add(title)
        if len(results) == top_k:
            break
    return results

@app.get("/hello")
def read_hello():
    return {"message": "Hello, world!"}

@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):
    results = search(query)
    return results

def run():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
