import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import re
from langdetect import detect, LangDetectException


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
    
    df = pd.read_csv('dataset/scraped_lyrics.csv')

   
    df['Cleaned Lyrics'] = df['Lyrics'].astype(str).apply(clean_text)

    
    df = df[df['Cleaned Lyrics'].str.len() > 20]

    
    df['Language'] = df['Cleaned Lyrics'].apply(detect_language)
    df = df[df['Language'] == 'en']  

   
    df = df.reset_index(drop=True)

    
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    
    lyrics_list = df['Cleaned Lyrics'].tolist()
    print("Generating embeddings for lyrics...")
    embeddings = sbert_model.encode(lyrics_list, convert_to_numpy=True, show_progress_bar=True)

    np.save('embeddings/lyrics_embeddings.npy', embeddings)
    df.to_csv('dataset/filtered_lyrics.csv', index=False)
    print("Embeddings and filtered lyrics saved successfully.")

if __name__ == "__main__":
    main()
