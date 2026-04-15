# =========================
# FAST SONG SEARCH PIPELINE
# =========================
# แยกเป็น 2 โหมด:
# 1) build_index() -> รันครั้งเดียว
# 2) load_and_search() -> ใช้งานเร็ว

import os
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------------
# CONFIG
# -------------------------
EMBEDDING_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"
DATA_PATH = "df_chunks.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
    import kagglehub

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

# =========================
# LOAD (FAST)
# =========================
def load_system():
    print("Loading cached data...")

    df_chunks = pd.read_csv(DATA_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL)

    return df_chunks, index, model, reranker

# =========================
# SEARCH
# =========================
'''
def search(query, df_chunks, index, model, reranker, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, 50)

    candidates = df_chunks.iloc[indices[0]].copy()
    pairs = [(query, row['combined_text']) for _, row in candidates.iterrows()]

    rerank_scores = reranker.predict(pairs)
    candidates['score'] = rerank_scores

    results = candidates.sort_values("score", ascending=False).head(top_k)

    for i, row in results.iterrows():
        print(f"{row['song']} - {row['artist']}")
        print(f"{row['chunks'][:200]}...")
        print("-"*50)
'''

def keyword_overlap_score(query, text):
    import re
    q_words = set(re.findall(r'\w+', query.lower()))
    t_words = set(re.findall(r'\w+', str(text).lower()))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def search(query, df_chunks, index, model, reranker, top_k=5, fetch_k=50):

    # -------------------------
    # 1. Retrieve
    # -------------------------
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, fetch_k)

    candidates = df_chunks.iloc[indices[0]].copy()
    candidates["retrieval_score"] = scores[0]

    # -------------------------
    # 2. Rerank
    # -------------------------
    pairs = []

    for _, row in candidates.iterrows():
        doc_text = (
            f"Song title: {row['song']}. "
            f"Artist: {row['artist']}. "
            f"Genre: {row['genre']}. "
            f"Emotion: {row['emotion']}. "
            f"Lyrics: {row['chunks']}"
        )
        pairs.append((query, doc_text))

    rerank_scores = reranker.predict(pairs)
    candidates["rerank_score"] = rerank_scores

    # sigmoid
    candidates["rerank_score_sigmoid"] = 1 / (1 + np.exp(-candidates["rerank_score"]))

    # -------------------------
    # 3. Keyword score
    # -------------------------
    candidates["keyword_score"] = candidates["combined_text"].apply(
        lambda x: keyword_overlap_score(query, x)
    )

    # -------------------------
    # 4. Normalize retrieval
    # -------------------------
    r_min = candidates["retrieval_score"].min()
    r_max = candidates["retrieval_score"].max()

    if r_max > r_min:
        candidates["retrieval_norm"] = (candidates["retrieval_score"] - r_min) / (r_max - r_min)
    else:
        candidates["retrieval_norm"] = 0.0

    # -------------------------
    # 5. Normalize popularity
    # -------------------------
    if "Popularity" in candidates.columns:
        pop_min = candidates["Popularity"].min()
        pop_max = candidates["Popularity"].max()

        if pop_max > pop_min:
            candidates["popularity_norm"] = (candidates["Popularity"] - pop_min) / (pop_max - pop_min)
        else:
            candidates["popularity_norm"] = 0.0
    else:
        candidates["popularity_norm"] = 0.0

    # -------------------------
    # 6. Final chunk score
    # -------------------------
    candidates["chunk_score"] = (
        0.50 * candidates["retrieval_norm"] +
        0.30 * candidates["rerank_score_sigmoid"] +
        0.15 * candidates["keyword_score"] +
        0.05 * candidates["popularity_norm"]
    )

    # -------------------------
    # 7. Deduplicate (สำคัญ!)
    # -------------------------
    best_idx = candidates.groupby(["song", "artist"])["chunk_score"].idxmax()
    song_level = candidates.loc[best_idx].copy()

    # -------------------------
    # 8. Sort
    # -------------------------
    results = song_level.sort_values("chunk_score", ascending=False).head(top_k)

    # -------------------------
    # 9. Show results
    # -------------------------
    print(f"\nQuery: {query}")
    print("=" * 80)

    for i, row in results.reset_index(drop=True).iterrows():
        print(f"{i+1}. {row['song']} - {row['artist']}")
        print(f"   Genre: {row.get('genre', '-')}")
        print(f"   Emotion: {row.get('emotion', '-')}")
        print(f"   Score: {row['chunk_score']:.4f}")
        print(f"   Preview: {str(row['lyrics_raw'])[:200]}...")
        print("-" * 80)

# =========================
# USAGE
# =========================
if __name__ == "__main__":

    #  ครั้งแรก (รันทีเดียว)
    # build_index()

    #  ใช้งานจริง
    df_chunks, index, model, reranker = load_system()

    while True:
        q = input("Search: ")
        search(q, df_chunks, index, model, reranker)
