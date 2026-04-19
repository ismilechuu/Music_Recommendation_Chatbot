import sys
import os

# Ensure project root is in the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import DATA_PATH, INDEX_PATH, MODEL_NAME, RERANK_MODEL

# =========================
# LOAD (FAST)
# =========================
@st.cache_resource(show_spinner=False)
def load_system():
    print("Loading cached data...")

    df_chunks = pd.read_csv(DATA_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL)

    return df_chunks, index, model, reranker

# =========================
# SCORE HELPERS
# =========================
def keyword_overlap_score(query, text):
    import re
    q_words = set(re.findall(r'\w+', query.lower()))
    t_words = set(re.findall(r'\w+', str(text).lower()))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)

def search(query, df_chunks, index, model, reranker, top_k=5, fetch_k=30, chat_history=None):

    # สร้าง Contextual Search Query (เอาคำถามเก่ามาต่อกับคำถามใหม่)
    search_query = query
    if chat_history and len(chat_history) > 0:
        # เพื่อไม่ให้ Query ยาวเกินไป เราอาจจะดึงแค่ 2 คำถามล่าสุดมาต่อ
        past_queries = [turn["query"] for turn in chat_history[-2:]]
        search_query = " ".join(past_queries) + " " + query

    # 1. Retrieve (ใช้ search_query ที่มีบริบทอดีตด้วย)
    query_vec = model.encode([search_query], normalize_embeddings=True)
    scores, indices = index.search(query_vec, fetch_k)

    candidates = df_chunks.iloc[indices[0]].copy()
    candidates["retrieval_score"] = scores[0]

    # 2. Rerank
    pairs = []
    for _, row in candidates.iterrows():
        doc_text = (
            f"Song title: {row['song']}. "
            f"Artist: {row['artist']}. "
            f"Genre: {row['genre']}. "
            f"Emotion: {row['emotion']}. "
            f"Lyrics: {row['chunks']}"
        )
        # เทียบกับ search_query
        pairs.append((search_query, doc_text))

    rerank_scores = reranker.predict(pairs)
    candidates["rerank_score"] = rerank_scores

    # sigmoid
    candidates["rerank_score_sigmoid"] = 1 / (1 + np.exp(-candidates["rerank_score"]))

    # 3. Keyword score
    candidates["keyword_score"] = candidates["combined_text"].apply(
        lambda x: keyword_overlap_score(search_query, x)
    )

    # 4. Normalize retrieval
    r_min = candidates["retrieval_score"].min()
    r_max = candidates["retrieval_score"].max()

    if r_max > r_min:
        candidates["retrieval_norm"] = (candidates["retrieval_score"] - r_min) / (r_max - r_min)
    else:
        candidates["retrieval_norm"] = 0.0

    # 5. Normalize popularity
    if "Popularity" in candidates.columns:
        pop_min = candidates["Popularity"].min()
        pop_max = candidates["Popularity"].max()

        if pop_max > pop_min:
            candidates["popularity_norm"] = (candidates["Popularity"] - pop_min) / (pop_max - pop_min)
        else:
            candidates["popularity_norm"] = 0.0
    else:
        candidates["popularity_norm"] = 0.0

    # 6. Final chunk score
    candidates["chunk_score"] = (
        0.50 * candidates["retrieval_norm"] +
        0.30 * candidates["rerank_score_sigmoid"] +
        0.15 * candidates["keyword_score"] +
        0.05 * candidates["popularity_norm"]
    )

    # 7. Deduplicate
    best_idx = candidates.groupby(["song", "artist"])["chunk_score"].idxmax()
    song_level = candidates.loc[best_idx].copy()

    # 8. Sort
    results = song_level.sort_values("chunk_score", ascending=False).head(top_k)

    return results
