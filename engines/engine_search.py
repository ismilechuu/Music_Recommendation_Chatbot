import os
import re
import sys
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Make project root importable so this module can load config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, INDEX_PATH, MODEL_NAME, RERANK_MODEL


# ==========================================================
# SEARCH CONFIGURATION
# ==========================================================
# These weights control how much each signal affects the final score.
# Retrieval signals decide general relevance.
# Mood/theme signals decide whether the song matches the requested vibe.
# Popularity is only a light tie-breaker.
DENSE_RETRIEVAL_WEIGHT = 0.25
SPARSE_RETRIEVAL_WEIGHT = 0.18
RERANK_WEIGHT = 0.20
KEYWORD_WEIGHT = 0.08
MOOD_WEIGHT = 0.10
THEME_WEIGHT = 0.14
POPULARITY_WEIGHT = 0.05

# Default retrieval limits.
DEFAULT_FETCH_K_DENSE = 50
DEFAULT_FETCH_K_SPARSE = 50
DEFAULT_TOP_K = 5

# Short follow-up words that usually mean
# “keep the previous request, but refine it”.
FOLLOWUP_HINTS = {
    "more", "less", "warmer", "softer", "gentler", "calmer", "sadder",
    "happier", "upbeat", "slower", "faster", "different", "another",
    "similar", "same", "lighter", "darker", "stronger", "weaker",
    "acoustic", "stripped", "mellower", "mellow", "soothing",
}

# Expansion terms help convert short follow-ups into richer retrieval queries.
# Example: “softer” becomes “soft gentle mellow tender ...”.
MODIFIER_EXPANSIONS: Dict[str, List[str]] = {
    "softer": ["soft", "gentle", "mellow", "tender", "acoustic", "intimate", "calm", "stripped down"],
    "gentler": ["gentle", "soft", "tender", "calm", "mellow", "intimate"],
    "warmer": ["warm", "comforting", "cozy", "gentle", "tender", "soft"],
    "calmer": ["calm", "peaceful", "gentle", "soft", "mellow", "soothing"],
    "sadder": ["sad", "melancholic", "heartbroken", "longing", "emotional"],
    "upbeat": ["upbeat", "energetic", "bright", "danceable", "lively"],
    "acoustic": ["acoustic", "unplugged", "guitar", "piano", "stripped down", "bare", "minimal", "intimate", "soft vocals"],
    "different": [],
    "another": [],
    "similar": [],
}

# Mood groups describe broad emotional direction.
# “positive” means desirable terms for that mood.
# “negative” means terms that should be penalized.
MOOD_GROUPS: Dict[str, Dict[str, Set[str]]] = {
    "warm": {
        "positive": {"warm", "comforting", "cozy", "gentle", "soft", "tender", "peaceful", "romantic", "calm"},
        "negative": {"anger", "angry", "rage", "aggressive", "hostile", "violent", "intense"},
    },
    "soft": {
        "positive": {"soft", "gentle", "mellow", "tender", "acoustic", "calm", "intimate", "peaceful", "subtle"},
        "negative": {"aggressive", "rage", "intense", "loud", "hostile", "explosive", "dramatic", "powerful"},
    },
    "sad": {
        "positive": {"sad", "melancholic", "heartbroken", "longing", "lonely", "wistful", "emotional"},
        "negative": {"party", "joyful", "celebration", "rage", "aggressive"},
    },
    "romantic": {
        "positive": {"romantic", "love", "tender", "warm", "soft", "intimate"},
        "negative": {"rage", "hostile", "aggressive", "cold"},
    },
    "upbeat": {
        "positive": {"upbeat", "energetic", "bright", "dance", "happy", "lively"},
        "negative": {"sad", "lonely", "slow", "melancholic"},
    },
}

# Theme groups describe more specific intent than mood.
# This helps separate requests like:
# - sad breakup song
# - sad song about missing someone
# - healing after heartbreak
THEME_GROUPS: Dict[str, Dict[str, Set[str]]] = {
    "missing_someone": {
        "triggers": {
            "missing someone", "miss someone", "i miss you", "missing you", "longing",
            "wish you were here", "without you", "far away", "gone", "absence", "waiting for you",
        },
        "positive": {
            "miss you", "missing you", "i miss you", "miss me", "longing", "yearning",
            "wish you were here", "without you", "gone", "far away", "come back", "waiting",
            "absence", "apart", "not here", "still see you", "remember you",
        },
        "negative": {"party", "club", "flex", "money", "brag", "fight", "revenge"},
    },
    "breakup": {
        "triggers": {"breakup", "break up", "broke up", "ex", "after we broke up"},
        "positive": {
            "breakup", "break up", "broke up", "ex", "we ended", "left me", "goodbye",
            "moved on", "over us", "ended", "split", "former lover",
        },
        "negative": {"party", "club", "celebration"},
    },
    "healing": {
        "triggers": {
            "healing", "healing song", "moving on", "move on", "recover", "recovery",
            "cope", "letting go", "let go", "closure", "after heartbreak",
            "after a breakup", "getting better", "starting over", "new beginning"
        },
        "positive": {
            "heal", "healing", "recover", "recovery", "moving on", "move on",
            "let go", "letting go", "cope", "closure", "hope", "hopeful",
            "stronger", "better", "growth", "peace", "peaceful", "mend",
            "getting better", "start again", "starting over", "new beginning",
            "after the pain", "rise again", "carry on", "find myself",
            "learn to live", "release", "freedom", "acceptance"
        },
        "negative": {
            "revenge", "rage", "violent", "obsessed", "can't let go", "cannot let go",
            "not over you", "still love you", "still need you", "beg you",
            "come back", "take me back", "why did you leave", "broken inside",
            "destroyed", "fall apart", "can't move on", "cannot move on"},
    },
    "loneliness": {
        "triggers": {"lonely", "loneliness", "alone", "by myself", "isolated"},
        "positive": {"lonely", "loneliness", "alone", "by myself", "isolated", "empty room", "nobody", "on my own", "no one"},
        "negative": {"party", "crowd", "celebration"},
    },
}

# Global TF-IDF objects are cached after load_system() runs once.
TFIDF_VECTORIZER = None
TFIDF_MATRIX = None


# ==========================================================
# LOAD MODELS AND DATA
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_system():
    """
    Load all resources needed for retrieval and reranking.

    Returns:
        df_chunks: DataFrame containing chunk-level song data
        index: FAISS dense retrieval index
        model: SentenceTransformer embedding model
        reranker: CrossEncoder reranking model
    """
    global TFIDF_VECTORIZER, TFIDF_MATRIX

    print("Loading cached data...")
    df_chunks = pd.read_csv(DATA_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL)

    # Build sparse retrieval index from combined text.
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        df_chunks["combined_text"].fillna("").astype(str).tolist()
    )

    TFIDF_VECTORIZER = tfidf_vectorizer
    TFIDF_MATRIX = tfidf_matrix
    return df_chunks, index, model, reranker


# ==========================================================
# BASIC TEXT UTILITIES
# ==========================================================
def normalize_text(text: str) -> str:
    """Normalize text for matching and comparison."""
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Split text into simple lowercase word tokens."""
    return re.findall(r"[a-zA-Z0-9']+", normalize_text(text))


def keyword_overlap_score(query: str, text: str) -> float:
    """Measure how many query tokens appear in the candidate text."""
    query_words = set(tokenize(query))
    text_words = set(tokenize(text))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert raw reranker logits into bounded 0-1 style scores."""
    return 1 / (1 + np.exp(-values))


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series. Return zeros if flat."""
    series_min = float(series.min())
    series_max = float(series.max())
    if series_max > series_min:
        return (series - series_min) / (series_max - series_min)
    return pd.Series(np.zeros(len(series)), index=series.index)


# ==========================================================
# QUERY UNDERSTANDING
# ==========================================================
def is_short_refinement(query: str) -> bool:
    """
    Detect whether the current query is a short follow-up refinement.

    Examples:
        - "something softer"
        - "more mellow"
        - "warmer"
        - "another one"
    """
    tokens = tokenize(query)
    if len(tokens) <= 5 and any(token in FOLLOWUP_HINTS for token in tokens):
        return True

    lowered = normalize_text(query)
    patterns = [
        r"\bi want (something )?(softer|warmer|gentler|calmer|sadder|happier)\b",
        r"\bmore\b",
        r"\bless\b",
        r"\banother\b",
        r"\bsimilar\b",
        r"\bdifferent\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def get_last_user_query(chat_history=None) -> str:
    """Return the latest user query from chat history, if available."""
    if not chat_history:
        return ""

    for turn in reversed(chat_history):
        query = turn.get("query")
        if query:
            return str(query)
    return ""


def extract_followup_modifiers(query: str) -> List[str]:
    """Extract refinement words like softer / warmer / acoustic from a follow-up query."""
    lowered = normalize_text(query)
    modifiers: List[str] = []

    for modifier in MODIFIER_EXPANSIONS.keys():
        if modifier in lowered:
            modifiers.append(modifier)

    for token in tokenize(query):
        if token in FOLLOWUP_HINTS and token not in modifiers:
            modifiers.append(token)

    return modifiers


def expand_modifiers(modifiers: List[str]) -> List[str]:
    """Expand short modifiers into richer retrieval vocabulary."""
    expanded_terms: List[str] = []
    for modifier in modifiers:
        expanded_terms.append(modifier)
        expanded_terms.extend(MODIFIER_EXPANSIONS.get(modifier, []))

    # Remove duplicates while preserving order.
    seen = set()
    ordered_terms = []
    for term in expanded_terms:
        if term not in seen:
            ordered_terms.append(term)
            seen.add(term)
    return ordered_terms


def build_search_query(query: str, chat_history=None) -> Tuple[str, bool]:
    """
    Build the actual retrieval query.

    If the user writes a short refinement, combine that refinement with the
    previous full user query so retrieval keeps the original intent.

    Returns:
        search_query: final query used for retrieval/reranking
        is_followup_rewrite: whether a rewrite happened
    """
    current_query = query.strip()
    last_query = get_last_user_query(chat_history)

    if last_query and is_short_refinement(current_query):
        modifiers = extract_followup_modifiers(current_query)
        expanded_terms = expand_modifiers(modifiers)
        modifier_text = " ".join(expanded_terms).strip()
        rewritten_query = f"{modifier_text} {last_query}".strip() if modifier_text else f"{current_query} {last_query}".strip()
        return rewritten_query, True

    return current_query, False


def extract_mood_preferences(query: str) -> Dict[str, Set[str]]:
    """
    Extract desired and undesired mood signals from the query.

    Returns a dictionary with:
        positive: terms that should be rewarded
        negative: terms that should be penalized
    """
    lowered = normalize_text(query)
    positive_terms: Set[str] = set()
    negative_terms: Set[str] = set()

    for mood_name, rules in MOOD_GROUPS.items():
        if mood_name in lowered or any(word in lowered for word in rules["positive"]):
            positive_terms.update(rules["positive"])
            negative_terms.update(rules["negative"])

    for modifier in extract_followup_modifiers(query):
        positive_terms.update(MODIFIER_EXPANSIONS.get(modifier, []))

        if modifier in {"softer", "gentler", "calmer", "warmer"}:
            negative_terms.update({"aggressive", "intense", "loud", "dramatic", "hostile", "rage", "powerful"})
        elif modifier == "sadder":
            negative_terms.update({"happy", "bright", "party", "upbeat"})
        elif modifier == "upbeat":
            negative_terms.update({"melancholic", "sad", "lonely", "slow"})

    return {"positive": positive_terms, "negative": negative_terms}


def extract_theme_preferences(query: str) -> Dict[str, Set[str]]:
    """
    Extract specific theme intent from the query.

    Themes are more specific than mood.
    Example: “missing someone” is a theme, while “sad” is a mood.
    """
    lowered = normalize_text(query)
    active_themes: Set[str] = set()
    positive_terms: Set[str] = set()
    negative_terms: Set[str] = set()

    for theme_name, rules in THEME_GROUPS.items():
        if any(trigger in lowered for trigger in rules["triggers"]):
            active_themes.add(theme_name)
            positive_terms.update(rules["positive"])
            negative_terms.update(rules["negative"])

    # Extra lightweight rule for common “missing someone” phrasing.
    if ("miss" in lowered and ("someone" in lowered or "you" in lowered)) or "missing someone" in lowered:
        active_themes.add("missing_someone")
        positive_terms.update(THEME_GROUPS["missing_someone"]["positive"])
        negative_terms.update(THEME_GROUPS["missing_someone"]["negative"])

    if "after heartbreak" in lowered or "after a breakup" in lowered:
        active_themes.add("healing")
        positive_terms.update(THEME_GROUPS["healing"]["positive"])
        negative_terms.update(THEME_GROUPS["healing"]["negative"])

    return {
        "active_themes": active_themes,
        "positive": positive_terms,
        "negative": negative_terms,
    }


# ==========================================================
# RETRIEVAL
# ==========================================================
def dense_retrieve(search_query: str, index, model, fetch_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve candidates with dense embeddings + FAISS."""
    query_vector = model.encode([search_query], normalize_embeddings=True)
    scores, indices = index.search(query_vector, fetch_k)
    return scores[0], indices[0]


def sparse_retrieve(search_query: str, fetch_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve candidates with sparse TF-IDF keyword matching."""
    if TFIDF_VECTORIZER is None or TFIDF_MATRIX is None:
        raise RuntimeError("TF-IDF resources are not initialized. Call load_system() first.")

    query_vector = TFIDF_VECTORIZER.transform([search_query])
    sparse_scores = linear_kernel(query_vector, TFIDF_MATRIX).flatten()
    top_indices = np.argsort(sparse_scores)[::-1][:fetch_k]
    top_scores = sparse_scores[top_indices]
    return top_scores, top_indices


def merge_dense_sparse_candidates(
    df_chunks: pd.DataFrame,
    dense_scores: np.ndarray,
    dense_indices: np.ndarray,
    sparse_scores: np.ndarray,
    sparse_indices: np.ndarray,
) -> pd.DataFrame:
    """
    Merge dense and sparse candidates into one table.

    Each candidate keeps both score types so later scoring can combine them.
    """
    dense_score_map = {
        int(idx): float(score)
        for idx, score in zip(dense_indices, dense_scores)
        if int(idx) >= 0
    }
    sparse_score_map = {
        int(idx): float(score)
        for idx, score in zip(sparse_indices, sparse_scores)
        if int(idx) >= 0
    }

    union_indices = list(dict.fromkeys(list(dense_score_map.keys()) + list(sparse_score_map.keys())))
    candidates = df_chunks.iloc[union_indices].copy()
    candidates["dense_retrieval_score"] = [dense_score_map.get(int(i), 0.0) for i in union_indices]
    candidates["sparse_retrieval_score"] = [sparse_score_map.get(int(i), 0.0) for i in union_indices]
    return candidates


def build_rerank_document_text(row: pd.Series) -> str:
    """Build the text representation passed to the cross-encoder reranker."""
    return (
        f"Song title: {row.get('song', '')}. "
        f"Artist: {row.get('artist', '')}. "
        f"Genre: {row.get('genre', '')}. "
        f"Emotion: {row.get('emotion', '')}. "
        f"Lyrics: {row.get('chunks', '')}"
    )


def add_rerank_scores(candidates: pd.DataFrame, search_query: str, reranker) -> pd.DataFrame:
    """Run cross-encoder reranking on the merged candidate set."""
    pairs = [(search_query, build_rerank_document_text(row)) for _, row in candidates.iterrows()]
    rerank_scores = reranker.predict(pairs)
    candidates["rerank_score"] = rerank_scores
    candidates["rerank_score_sigmoid"] = sigmoid(candidates["rerank_score"].values)
    return candidates


def add_keyword_scores(candidates: pd.DataFrame, search_query: str) -> pd.DataFrame:
    """Add simple token-overlap scores as a lightweight lexical signal."""
    candidates["keyword_score"] = candidates["combined_text"].apply(
        lambda value: keyword_overlap_score(search_query, str(value))
    )
    return candidates


# ==========================================================
# MOOD AND THEME MATCHING
# ==========================================================
def build_candidate_haystack(row: pd.Series) -> str:
    """
    Build one searchable text blob from the main song fields.

    This is used by mood/theme rule matching so we can check important words
    across song title, artist, emotion, genre, and a shortened combined text.
    """
    return " ".join(
        [
            str(row.get("song", "")),
            str(row.get("artist", "")),
            str(row.get("emotion", "")),
            str(row.get("genre", "")),
            str(row.get("combined_text", ""))[:800],
        ]
    ).lower()


def mood_match_score(preferences: Dict[str, Set[str]], row: pd.Series) -> float:
    """Reward candidates that contain desired mood vocabulary."""
    positive_terms = preferences["positive"]
    if not positive_terms:
        return 0.0

    haystack = build_candidate_haystack(row)
    matches = sum(1 for term in positive_terms if term in haystack)
    return min(1.0, matches / max(3, len(positive_terms)))


def mood_conflict_penalty(preferences: Dict[str, Set[str]], row: pd.Series) -> float:
    """Penalize candidates that contain conflicting mood vocabulary."""
    negative_terms = preferences["negative"]
    if not negative_terms:
        return 0.0

    haystack = build_candidate_haystack(row)
    conflicts = sum(1 for term in negative_terms if term in haystack)

    if conflicts == 0:
        return 0.0
    if conflicts == 1:
        return 0.08
    if conflicts == 2:
        return 0.16
    return 0.24


def add_mood_scores(candidates: pd.DataFrame, preferences: Dict[str, Set[str]]) -> pd.DataFrame:
    """Add mood match and mood penalty signals to candidates."""
    candidates["mood_match_score"] = candidates.apply(lambda row: mood_match_score(preferences, row), axis=1)
    candidates["mood_penalty"] = candidates.apply(lambda row: mood_conflict_penalty(preferences, row), axis=1)
    return candidates


def theme_match_score(theme_preferences: Dict[str, Set[str]], row: pd.Series) -> float:
    """Reward candidates that match the requested theme."""
    positive_terms = theme_preferences["positive"]
    if not positive_terms:
        return 0.0

    haystack = build_candidate_haystack(row)
    matches = sum(1 for term in positive_terms if term in haystack)
    return min(1.0, matches / max(2, len(positive_terms) * 0.35))


def theme_conflict_penalty(theme_preferences: Dict[str, Set[str]], row: pd.Series) -> float:
    """Penalize candidates that clash with the requested theme."""
    negative_terms = theme_preferences["negative"]
    if not negative_terms:
        return 0.0

    haystack = build_candidate_haystack(row)
    conflicts = sum(1 for term in negative_terms if term in haystack)

    if conflicts == 0:
        return 0.0
    if conflicts == 1:
        return 0.06
    return 0.12


def add_theme_scores(candidates: pd.DataFrame, theme_preferences: Dict[str, Set[str]]) -> pd.DataFrame:
    """Add theme match and theme penalty signals to candidates."""
    candidates["theme_match_score"] = candidates.apply(lambda row: theme_match_score(theme_preferences, row), axis=1)
    candidates["theme_penalty"] = candidates.apply(lambda row: theme_conflict_penalty(theme_preferences, row), axis=1)
    return candidates


def apply_theme_hard_filter(candidates: pd.DataFrame, theme_preferences: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Apply a stricter filter only for highly specific themes.

    We do not hard-filter every theme, because broad themes could become too narrow.
    For now, this stricter filter is used only for “missing someone”.
    """
    active_themes = theme_preferences.get("active_themes", set())
    if not active_themes:
        return candidates

    if "missing_someone" in active_themes:
        filtered = candidates[candidates["theme_match_score"] >= 0.12].copy()
        if len(filtered) >= 6:
            return filtered

    return candidates


# ==========================================================
# DUPLICATE CONTROL
# ==========================================================
def collect_previously_recommended_songs(chat_history=None) -> Set[Tuple[str, str]]:
    """Collect all (song, artist) pairs that were already recommended earlier."""
    seen_pairs: Set[Tuple[str, str]] = set()
    if not chat_history:
        return seen_pairs

    for turn in chat_history:
        results = turn.get("results")
        if results is None:
            continue

        if isinstance(results, pd.DataFrame):
            result_df = results
        else:
            try:
                result_df = pd.DataFrame(results)
            except Exception:
                continue

        for _, row in result_df.iterrows():
            song = normalize_text(row.get("song", ""))
            artist = normalize_text(row.get("artist", ""))
            if song and artist:
                seen_pairs.add((song, artist))

    return seen_pairs


def exclude_previously_recommended(candidates: pd.DataFrame, chat_history=None) -> pd.DataFrame:
    """
    Remove songs that were already recommended in earlier turns.

    If filtering removes everything, return the original candidates as a fallback
    so the system does not end up with no result at all.
    """
    seen_pairs = collect_previously_recommended_songs(chat_history)
    if not seen_pairs:
        return candidates

    mask = candidates.apply(
        lambda row: (normalize_text(row.get("song", "")), normalize_text(row.get("artist", ""))) not in seen_pairs,
        axis=1,
    )
    filtered_candidates = candidates.loc[mask].copy()
    if filtered_candidates.empty:
        return candidates
    return filtered_candidates


# ==========================================================
# FINAL SCORING AND RANKING
# ==========================================================
def add_normalized_feature_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    """Normalize retrieval and popularity signals before final score fusion."""
    candidates["dense_retrieval_norm"] = normalize_series(candidates["dense_retrieval_score"])
    candidates["sparse_retrieval_norm"] = normalize_series(candidates["sparse_retrieval_score"])

    if "Popularity" in candidates.columns:
        candidates["popularity_norm"] = normalize_series(candidates["Popularity"].fillna(0))
    else:
        candidates["popularity_norm"] = 0.0

    return candidates


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all signals into one final chunk-level score.

    - Dense/sparse/rerank handle retrieval relevance.
    - Mood/theme handle intent alignment.
    - Popularity is a weak bonus.
    - Penalties correct mismatched candidates.
    """
    candidates["chunk_score"] = (
        DENSE_RETRIEVAL_WEIGHT * candidates["dense_retrieval_norm"]
        + SPARSE_RETRIEVAL_WEIGHT * candidates["sparse_retrieval_norm"]
        + RERANK_WEIGHT * candidates["rerank_score_sigmoid"]
        + KEYWORD_WEIGHT * candidates["keyword_score"]
        + MOOD_WEIGHT * candidates["mood_match_score"]
        + THEME_WEIGHT * candidates["theme_match_score"]
        + POPULARITY_WEIGHT * candidates["popularity_norm"]
        - candidates["mood_penalty"]
        - candidates["theme_penalty"]
    )
    return candidates


def rank_final_song_results(candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Collapse chunk-level candidates into song-level results.

    A song may have multiple lyric chunks. This keeps only the best-scoring
    chunk per (song, artist) pair, then returns the top songs.
    """
    best_indices = candidates.groupby(["song", "artist"])["chunk_score"].idxmax()
    song_level = candidates.loc[best_indices].copy()
    return song_level.sort_values("chunk_score", ascending=False).head(top_k)


# ==========================================================
# MAIN SEARCH PIPELINE
# ==========================================================
def search(query, df_chunks, index, model, reranker, top_k=DEFAULT_TOP_K, fetch_k=70, chat_history=None):
    """
    Main retrieval pipeline used by the app.

    Steps:
        1. Build final search query
        2. Extract mood and theme preferences
        3. Retrieve dense and sparse candidates
        4. Merge candidates and remove duplicates from prior turns
        5. Rerank candidates
        6. Add lexical, mood, and theme signals
        7. Apply theme hard filter when needed
        8. Normalize scores and compute final score
        9. Collapse to song-level results
    """
    search_query, is_followup_rewrite = build_search_query(query, chat_history)
    mood_preferences = extract_mood_preferences(search_query)
    theme_preferences = extract_theme_preferences(search_query)

    dense_scores, dense_indices = dense_retrieve(
        search_query,
        index,
        model,
        min(fetch_k, DEFAULT_FETCH_K_DENSE),
    )
    sparse_scores, sparse_indices = sparse_retrieve(
        search_query,
        min(fetch_k, DEFAULT_FETCH_K_SPARSE),
    )

    candidates = merge_dense_sparse_candidates(
        df_chunks,
        dense_scores,
        dense_indices,
        sparse_scores,
        sparse_indices,
    )
    candidates = exclude_previously_recommended(candidates, chat_history)
    candidates = add_rerank_scores(candidates, search_query, reranker)
    candidates = add_keyword_scores(candidates, search_query)
    candidates = add_mood_scores(candidates, mood_preferences)
    candidates = add_theme_scores(candidates, theme_preferences)
    candidates = apply_theme_hard_filter(candidates, theme_preferences)
    candidates = add_normalized_feature_scores(candidates)
    candidates = score_candidates(candidates)

    results = rank_final_song_results(candidates, top_k=top_k)

    # Debug / inspection metadata to help verify retrieval behavior.
    results["search_query_used"] = search_query
    results["is_followup_rewrite"] = is_followup_rewrite
    results["active_themes"] = ", ".join(sorted(theme_preferences.get("active_themes", set())))
    return results
