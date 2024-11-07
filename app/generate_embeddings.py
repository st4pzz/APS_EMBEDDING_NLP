import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import re
from langdetect import detect, LangDetectException
from sklearn.metrics.pairwise import cosine_similarity
import os

def clean_text(text):
  
    text = text.lower()
  
    text = re.sub(r'[^a-z0-9\s]', '', text)
  
    text = re.sub(r'\s+', ' ', text)
   
    text = text.strip()
    return text

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def main():
  
    if os.path.exists('../dataset/filtered_lyrics.csv'):
        df = pd.read_csv('../dataset/filtered_lyrics.csv')
        print("Loaded filtered lyrics.")
    else:
        
        df = pd.read_csv('../dataset/scraped_lyrics.csv')
       
        df['Cleaned Lyrics'] = df['Lyrics'].astype(str).apply(clean_text)
        
        df = df[df['Cleaned Lyrics'].str.len() > 20]
      
        df['Language'] = df['Cleaned Lyrics'].apply(detect_language)
        df = df[df['Language'] == 'en']
       
        df = df.reset_index(drop=True)
       
        df.to_csv('../dataset/filtered_lyrics.csv', index=False)
        print("Processed and saved filtered lyrics.")

   
    if os.path.exists('../embedding/lyrics_embeddings.npy'):
        embeddings = np.load('../embedding/lyrics_embeddings.npy')
        print("Loaded embeddings from file.")
    else:
        
        sbert_model = SentenceTransformer('all-mpnet-base-v2')
      
        lyrics_list = df['Cleaned Lyrics'].tolist()
        
        print("Generating embeddings for lyrics...")
        embeddings = sbert_model.encode(lyrics_list, convert_to_numpy=True, show_progress_bar=True)
       
        os.makedirs('../embedding', exist_ok=True)
        np.save('../embedding/lyrics_embeddings.npy', embeddings)
        print("Embeddings saved successfully.")

   
    if 'sbert_model' not in locals():
        sbert_model = SentenceTransformer('all-mpnet-base-v2')

    def get_query_embedding(query):
        query = clean_text(query)
        return sbert_model.encode(query, convert_to_numpy=True)

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
                    'Similarity': similarities[idx],
                    'Lyrics': df.iloc[idx]['Lyrics']
                })
                titles.add(title)
            if len(results) == top_k:
                break
        return pd.DataFrame(results)

    
    print("Welcome to the lyrics search system!")
    while True:
        query = input("Enter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        results = search(query, top_k=10)
        print(f"Top {len(results)} results for query '{query}':")
        print(results[['Song Name','Similarity']])
        print("\n")

if __name__ == "__main__":
    main()
