import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.engine_search import load_system, search
from engines.engine_llm import recommend_songs


def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""


def add_chat_turn(user_query, recommendation, results, error_message=None):
    st.session_state.chat_history.append(
        {
            "query": user_query,
            "recommendation": recommendation,
            "results": results.copy() if results is not None else None,
            "error": error_message,
        }
    )


def render_match_card(rank, row):
    genre = str(row.get("genre", "Unknown")).replace("[", "").replace("]", "").replace("'", "")
    emotion = str(row.get("emotion", "Unknown"))

    st.markdown(
        f"""
        <div class="song-card">
            <div class="song-title">{rank}. {row['song']}</div>
            <div class="song-artist">by {row['artist']}</div>
            <div class="badge-row">
                <span class="badge">{genre}</span>
                <span class="badge">{emotion}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_grid(results):
    if results is None or len(results) == 0:
        st.markdown('<div class="empty-note">No matches found.</div>', unsafe_allow_html=True)
        return

    results = results.reset_index(drop=True)

    for i in range(0, len(results), 2):
        cols = st.columns(2, gap="medium")

        with cols[0]:
            render_match_card(i + 1, results.iloc[i])

        if i + 1 < len(results):
            with cols[1]:
                render_match_card(i + 2, results.iloc[i + 1])


def process_query(query, df_chunks, index, model, reranker):
    query = query.strip()
    if not query:
        return

    with st.spinner("Searching for the best semantic matches..."):
        results = search(query, df_chunks, index, model, reranker, chat_history=st.session_state.chat_history)

    recommendation = None
    error_message = None

    with st.spinner("Analyzing songs and generating recommendation via LLM..."):
        try:
            recommendation = recommend_songs(query, results, st.session_state.chat_history)
        except Exception as e:
            error_message = str(e)
            recommendation = (
                "I found some relevant songs for your request, but I couldn’t generate "
                "a full AI explanation right now."
            )

    add_chat_turn(query, recommendation, results, error_message=error_message)
    st.session_state.pending_query = ""


def main():
    st.set_page_config(page_title="SongSense AI", page_icon="🎵", layout="wide")
    init_state()

    st.markdown("""
        <style>
        .stApp {
            background: #07122b;
            color: #e7ecff;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 0.15rem;
            padding-bottom: 5rem;
        }

        header[data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        .top-search-wrap {
            position: sticky;
            top: 0;
            z-index: 999;
            background: rgba(7, 18, 43, 0.94);
            backdrop-filter: blur(10px);
            padding-top: 0.05rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 0.9rem;
        }

        .main-title {
            text-align: center;
            font-size: 3.2rem;
            background: -webkit-linear-gradient(45deg, #FF6B6B, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.18rem;
            margin-top: 0;
            letter-spacing: -0.02em;
        }

        .sub-text {
            text-align: center;
            color: #9aa0aa;
            font-size: 1.02rem;
            margin-bottom: 0.9rem;
        }

        .chat-user-wrap {
            display: flex;
            justify-content: flex-end;
            margin: 0.4rem 0 1.1rem 0;
        }

        .chat-user {
            max-width: 70%;
            background: rgba(139, 92, 246, 0.10);
            border: 1px solid rgba(139, 92, 246, 0.20);
            color: #f5f7ff;
            padding: 0.95rem 1.1rem;
            border-radius: 16px;
            text-align: right;
            line-height: 1.7;
            font-size: 1.02rem;
        }

        .chat-user-label {
            color: #a78bfa;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .turn-wrap {
            margin-bottom: 2rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .assistant-shell {
            display: flex;
            align-items: flex-start;
            gap: 0.9rem;
            margin-bottom: 0.8rem;
        }

        .assistant-icon {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: rgba(45, 52, 73, 0.78);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #f7b267;
            font-size: 1.1rem;
            flex-shrink: 0;
            margin-top: 0.15rem;
        }

        .assistant-main {
            flex: 1;
            min-width: 0;
        }

        .section-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.8rem;
        }

        .intro-text {
            color: #dfe6ff;
            font-size: 1.02rem;
            line-height: 1.7;
            margin-bottom: 0.85rem;
        }

        .recommendation-text {
            color: #eef2ff;
            font-size: 1.04rem;
            line-height: 1.95;
        }

        .recommendation-text p,
        .recommendation-text ul,
        .recommendation-text ol {
            margin-top: 0;
            margin-bottom: 1rem;
        }

        .recommendation-text li {
            margin-bottom: 0.45rem;
        }

        .soft-warning {
            background: rgba(120, 92, 246, 0.10);
            border: 1px solid rgba(139, 92, 246, 0.16);
            color: #d7cbff;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.9rem;
        }

        .matches-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffffff;
            margin-top: 1.1rem;
            margin-bottom: 0.9rem;
        }

        .song-card {
            background: rgba(20, 22, 34, 0.92);
            border: 1px solid rgba(255, 107, 107, 0.18);
            border-radius: 16px;
            padding: 1rem 1rem;
            min-height: 138px;
            margin-bottom: 1rem;
        }

        .song-title {
            color: #ff6b6b;
            font-weight: 800;
            font-size: 1.18rem;
            line-height: 1.3;
            margin-bottom: 0.38rem;
        }

        .song-artist {
            color: #c3c8d8;
            font-size: 0.98rem;
            margin-bottom: 0.85rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }

        .badge {
            background: rgba(78, 205, 196, 0.18);
            color: #4ECDC4;
            padding: 0.30rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            display: inline-block;
        }

        .empty-note {
            color: #9aa0aa;
        }

        /* search area */
        .stForm,
        form[data-testid="stForm"] {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            background: transparent !important;
        }

        .stTextInput > div,
        .stTextInput > div > div,
        div[data-testid="stTextInputRootElement"],
        div[data-testid="stTextInputRootElement"] > div {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }

        /* ตัวช่องพิมพ์จริง */
        .stTextInput input,
        .stTextInput input:focus,
        .stTextInput input:active {
            border-radius: 16px !important;
            background: rgba(28, 34, 54, 0.98) !important;
            color: #f7f8ff !important;
            border: 1.2px solid rgba(139, 92, 246, 0.20) !important;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18) !important;
            outline: none !important;
        }

        /* label */
        .stTextInput > label {
            color: #f1f3ff !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            border: none !important;
            border-radius: 14px !important;
            padding: 0.82rem 1.2rem !important;
            font-weight: 800 !important;
            color: white !important;
        }

        /* ปุ่มแรก = Search */
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22) !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.28) !important;
        }

        /* ปุ่มสอง = Clear Chat */
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stFormSubmitButton"] > button {
            background: rgba(45, 52, 73, 0.92) !important;
            color: #e7ecff !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            box-shadow: none !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stFormSubmitButton"] > button:hover {
            background: rgba(58, 66, 92, 0.96) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
        }

        @media (max-width: 900px) {
            .chat-user {
                max-width: 100%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models and vector index (this may take a moment on first load)..."):
        df_chunks, index, model, reranker = load_system()

    st.markdown('<div class="top-search-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">🎵 SongSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">Find the perfect song for any mood, powered by Vector Search &amp; LLMs.</div>',
        unsafe_allow_html=True
    )

    with st.form("search_form", clear_on_submit=True):
        query = st.text_input(
            "What kind of song are you looking for?",
            value=st.session_state.pending_query,
            placeholder="E.g., A sad song about missing someone on Christmas",
            key="chat_input_top"
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            search_clicked = st.form_submit_button("Search", use_container_width=True)
        with col_btn2:
            clear_clicked = st.form_submit_button("Clear Chat", use_container_width=False)

    st.markdown('</div>', unsafe_allow_html=True)

    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state.pending_query = ""
        st.rerun()

    if search_clicked:
        if query.strip():
            process_query(query, df_chunks, index, model, reranker)
            st.rerun()
        else:
            st.warning("Please enter a query first!")

    for turn in reversed(st.session_state.chat_history):
        st.markdown('<div class="turn-wrap">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="chat-user-wrap">
                <div class="chat-user">
                    <div class="chat-user-label">You</div>
                    <div>{turn['query']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="assistant-shell">
                <div class="assistant-icon">✨</div>
                <div class="assistant-main">
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">AI Music Recommendation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="intro-text">That’s a mood. I’ve curated a selection of tracks that best match the feeling and context of your request.</div>',
            unsafe_allow_html=True
        )

        if turn["error"]:
            st.markdown(
                '<div class="soft-warning">AI explanation is unavailable right now. Showing the best search results instead.</div>',
                unsafe_allow_html=True,
            )

        if turn["recommendation"]:
            st.markdown('<div class="recommendation-text">', unsafe_allow_html=True)
            st.markdown(turn["recommendation"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="matches-title">🎯 Top Matches</div>', unsafe_allow_html=True)
        render_match_grid(turn["results"])

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()