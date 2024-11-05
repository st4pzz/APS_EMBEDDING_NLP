from fastapi import FastAPI, Query, Response
import os
import uvicorn
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


import json

app = FastAPI()

df = pd.read_csv('../dataset/scraped_lyrics.csv')

enhanced_embeddings = np.load('../embedding/enhanced_embeddings.npy')

assert len(enhanced_embeddings) == len(df), "Embeddings and dataframe length mismatch."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()


input_dim = 768      
hidden_dim1 = 512
hidden_dim2 = 256

encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU()
).to(device)

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(DenoisingAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
       
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

full_model = DenoisingAutoEncoder(input_dim, hidden_dim1, hidden_dim2).to(device)
full_model.load_state_dict(torch.load('../embedding/autoencoder.pth', map_location=device))
full_model.eval()

encoder.load_state_dict(full_model.encoder.state_dict())
encoder.eval()


def get_query_embedding(query):
    with torch.no_grad():

        inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state  

        embedding = torch.mean(last_hidden_states, dim=1) 
        
        enhanced_embedding = encoder(embedding)
        return enhanced_embedding.cpu().numpy().squeeze()


def search(query, top_k=10):

    query_embedding = get_query_embedding(query)

    similarities = cosine_similarity([query_embedding], enhanced_embeddings)[0]

    top_k_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_k_indices]

    results = results.copy()
    results['Similarity'] = similarities[top_k_indices]
    return results

@app.get("/hello")
def read_hello():
    return {"message": "hello world"}

@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):



    results = search(query)

    results_json = results[['Song Name', 'Lyrics', 'Similarity']].to_json(orient='records')
    return Response(content=results_json, media_type="application/json")

def run():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
