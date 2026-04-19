import sys
import os

# Ensure project root is in the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import faiss
import re
import kagglehub
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_PATH, INDEX_PATH, DATA_PATH, MODEL_NAME

# -------------------------
# CLEAN FUNCTION
# -------------------------
def clean_lyrics(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'[^a-zA-Zก-๙0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------
# CHUNK
# -------------------------
def chunk_text(text, chunk_size=120, overlap=30, max_chunks=10):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if len(chunks) >= max_chunks:
            break
    return chunks

# =========================
# BUILD (RUN ONCE)
# =========================
def build_index():
    print("Loading dataset...")
    path = kagglehub.dataset_download("devdope/900k-spotify")
    df = pd.read_csv(os.path.join(path, "spotify_dataset.csv"))

    df = df.drop_duplicates().dropna(subset=['song'])
    df['lyrics_clean'] = df['text'].apply(clean_lyrics)

    print("Sampling...")
    df_top = df.sort_values(by='Popularity', ascending=False).head(4000)
    df_rand = df.drop(df_top.index).sample(n=4000, random_state=42)
    df_small = pd.concat([df_top, df_rand]).sample(frac=1, random_state=42)

    df_small['length'] = df_small['lyrics_clean'].apply(lambda x: len(x.split()))
    df_small = df_small[(df_small['length'] > 20) & (df_small['length'] < 2000)].reset_index(drop=True)

    print("Chunking...")
    df_small['chunks'] = df_small['lyrics_clean'].apply(chunk_text)
    df_chunks = df_small.explode('chunks').reset_index(drop=True)

    df_chunks = df_chunks.rename(columns={
        "Artist(s)": "artist",
        "Genre": "genre",
        "text": "lyrics_raw"
    })

    df_chunks["combined_text"] = (
        "Song title: " + df_chunks["song"].astype(str) + ". " +
        "Artist: " + df_chunks["artist"].astype(str) + ". " +
        "Genre: " + df_chunks["genre"].astype(str) + ". " +
        "Emotion: " + df_chunks["emotion"].astype(str) + ". " +
        "Lyrics: " + df_chunks["chunks"].astype(str)
    )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding (THIS TAKES TIME)...")
    embeddings = model.encode(
        df_chunks["combined_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("Saving files...")
    np.save(EMBEDDING_PATH, embeddings)
    faiss.write_index(index, INDEX_PATH)
    df_chunks.to_csv(DATA_PATH, index=False)

    print("DONE ✅")

if __name__ == "__main__":
    build_index()