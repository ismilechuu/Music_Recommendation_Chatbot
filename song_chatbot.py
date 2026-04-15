import kagglehub
import os
import pandas as pd
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np


path = kagglehub.dataset_download("devdope/900k-spotify")
df = pd.read_csv(os.path.join(path, "spotify_dataset.csv"))

duplicated_rows = df[df.duplicated(keep=False)]
df = df.drop_duplicates(keep='first')
df = df.dropna(subset=['song'])


### ลบเนื้อเพลงในบางส่วนที่ไม่มีความสำคัญ
def clean_lyrics(text):
    text = str(text).lower()

    # 1) ลบ [ ... ]
    text = re.sub(r'\[.*?\]', '', text)

    # 2) ลบ ( ... )
    text = re.sub(r'\(.*?\)', '', text)

    # 4) ลดคำซ้ำติดกัน
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # 5) ลบอักขระพิเศษ
    text = re.sub(r'[^a-zA-Zก-๙0-9\s]', '', text)

    # 6) ลบช่องว่างเกิน
    text = re.sub(r'\s+', ' ', text).strip()

    return text

### เพิ่ม column 'lyrics_clean'
df['lyrics_clean'] = df['text'].apply(clean_lyrics)

### ลดขนาด dataset โดยเลือกจากเพลงที่มีค่า Popularity 4000 เพลง
df_top = df.sort_values(by='Popularity', ascending=False).head(4000)

# และเลือกจากการสุ่มอีก 4000 เพลง
df_rest = df.drop(df_top.index)
df_rand = df_rest.sample(n=4000, random_state=42)

df_small = pd.concat([df_top, df_rand]).sample(frac=1, random_state=42)

### คำนวณจำนวนคำของเนื้อเพลงในทุก ๆ เพลง
df_small['length'] = df_small['lyrics_clean'].apply(lambda x: len(x.split()))

### ลบเพลงที่มีขนาดความยาวเนื้อเพลงผิดปกติ
df_small = df_small[(df_small['length'] > 20) & (df_small['length'] < 2000)].copy()
df_small = df_small.reset_index(drop=True)

### chunking

def chunk_text(text, chunk_size=120, overlap=30, max_chunks=10):
    text = text.replace("\n", " ")
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

### สร้าง column chunks
df_small['chunks'] = df_small['lyrics_clean'].apply(chunk_text)

### นับจำนวน chunk ที่สร้าง
df_small['num_chunks'] = df_small['chunks'].apply(len)

### Expand Chunks for Embedding
df_chunks = df_small.explode('chunks').reset_index(drop=True)



# rename columns
df_chunks = df_chunks.rename(columns={
    "Artist(s)": "artist",
    "Genre": "genre",
    "text": "lyrics_raw"
})

df_chunks["combined_text"] = (
    "Song title: " + df_chunks["song"].fillna("").astype(str) + ". " +
    "Artist: " + df_chunks["artist"].fillna("").astype(str) + ". " +
    "Genre: " + df_chunks["genre"].fillna("").astype(str) + ". " +
    "Emotion: " + df_chunks["emotion"].fillna("").astype(str) + ". " +
    " | Lyrics Chunk: " + df_chunks["chunks"].fillna("").astype(str).str[:800]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# create embeddings
song_embeddings = model.encode(
    df_chunks["combined_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# สร้าง FAISS index
dimension = song_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(song_embeddings)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ฟังก์ชันค้นหา
def retrieve_candidates(query, fetch_k=120):
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_vector, fetch_k)

    candidates = df_chunks.iloc[indices[0]].copy()
    candidates["retrieval_score"] = scores[0]
    candidates = candidates.sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    return candidates

def rerank_with_crossencoder(query, candidates):
    pairs = []

    for _, row in candidates.iterrows():
        doc_text = (
            f"Song title: {row['song']}. "
            f"Artist: {row['artist']}. "
            f"Genre: {row['genre']}. "
            f"Emotion: {row['emotion']}. "
            f"Lyrics excerpt: {row['chunks']}"
        )
        pairs.append((query, doc_text))

    rerank_scores = reranker.predict(pairs)
    candidates = candidates.copy()
    candidates["rerank_score"] = rerank_scores

    # ใช้ sigmoid
    candidates["rerank_score_sigmoid"] = 1 / (1 + np.exp(-candidates["rerank_score"]))

    return candidates

def keyword_overlap_score(query, text):
    q_words = set(re.findall(r'\w+', query.lower()))
    t_words = set(re.findall(r'\w+', str(text).lower()))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)

def search_unique_songs(query, top_k=5, fetch_k=50):
    # 1) retrieve
    candidates = retrieve_candidates(query, fetch_k=fetch_k)

    # 2) rerank
    candidates = rerank_with_crossencoder(query, candidates)

    # 3) keyword overlap
    candidates["keyword_score"] = candidates["combined_text"].apply(
        lambda x: keyword_overlap_score(query, x)
    )

    # 4) normalize retrieval score
    r_min = candidates["retrieval_score"].min()
    r_max = candidates["retrieval_score"].max()
    if r_max > r_min:
        candidates["retrieval_norm"] = (candidates["retrieval_score"] - r_min) / (r_max - r_min)
    else:
        candidates["retrieval_norm"] = 0.0

    # 5) normalize popularity
    pop_min = candidates["Popularity"].min()
    pop_max = candidates["Popularity"].max()
    if pop_max > pop_min:
        candidates["popularity_norm"] = (candidates["Popularity"] - pop_min) / (pop_max - pop_min)
    else:
        candidates["popularity_norm"] = 0.0

    # 6) score ระดับ chunk
    candidates["chunk_score"] = (
        0.50 * candidates["retrieval_norm"] +
        0.30 * candidates["rerank_score_sigmoid"] +
        0.15 * candidates["keyword_score"] +
        0.05 * candidates["popularity_norm"]
    )

    # 7) เลือก chunk ที่ดีที่สุดจริงของแต่ละเพลง
    best_idx = candidates.groupby(["song", "artist"])["chunk_score"].idxmax()
    song_level = candidates.loc[best_idx].copy()

    # 8) เรียงคะแนนระดับเพลง
    song_level = song_level.sort_values("chunk_score", ascending=False).reset_index(drop=True)

    # 9) ใช้ชื่อ final_score สำหรับแสดงผล
    song_level["final_score"] = song_level["chunk_score"]

    # 10) คืนผลลัพธ์
    return song_level.head(top_k)[[
        "song", "artist", "genre", "emotion", "Popularity",
        "lyrics_raw", "chunks",
        "retrieval_score", "rerank_score", "rerank_score_sigmoid",
        "keyword_score", "final_score"
    ]]

def show_unique_results(query, top_k=5, fetch_k=50):
    results = search_unique_songs(query, top_k=top_k, fetch_k=fetch_k)

    print(f"Query: {query}")
    print("-" * 90)

    for i, row in results.reset_index(drop=True).iterrows():
        print(f"{i+1}. {row['song']} - {row['artist']}")
        print(f"   Genre: {row['genre']}")
        print(f"   Emotion: {row['emotion']}")
        print(f"   Popularity: {row['Popularity']}")
        print(f"   Retrieval Score: {row['retrieval_score']:.4f}")
        print(f"   Rerank Score: {row['rerank_score']:.4f}")
        print(f"   rerank_score_sigmoid: {row['rerank_score_sigmoid']:.4f}")
        print(f"   Keyword Score: {row['keyword_score']:.4f}")
        print(f"   Final Score: {row['final_score']:.4f}")
        print(f"   Original Lyrics Preview: {str(row['lyrics_raw'])[:500]}...")
        print("-" * 90)

test_queries = [
    "songs about ex-love",
    "sad breakup songs",
    "warm christmas songs",
    "songs about missing someone"
]

for q in test_queries:
    print("=" * 100)
    show_unique_results(q, top_k=5, fetch_k=50)