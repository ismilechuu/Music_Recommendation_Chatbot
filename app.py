import streamlit as st
import sys
import os
import html
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.engine_search import load_system, search
from engines.engine_llm import recommend_songs


def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_chat_turn(user_query, recommendation, results, error_message=None):
    st.session_state.chat_history.append(
        {
            "query": user_query,
            "recommendation": recommendation,
            "results": results.copy() if results is not None else None,
            "error": error_message,
        }
    )


def render_html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def safe_text(value) -> str:
    return html.escape("" if value is None else str(value))


def format_recommendation_html(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    escaped = html.escape(text)
    escaped = escaped.replace("\n\n", "<br><br>")
    escaped = escaped.replace("\n", "<br>")
    return escaped


def run_chat_query(query, df_chunks, index, model, reranker):
    with st.spinner("Searching for the best matches..."):
        results = search(
            query,
            df_chunks,
            index,
            model,
            reranker,
            chat_history=st.session_state.chat_history,
        )

    recommendation = None
    error_message = None

    with st.spinner("Generating recommendation..."):
        try:
            recommendation = recommend_songs(query, results, st.session_state.chat_history)
        except Exception as e:
            error_message = str(e)
            recommendation = (
                "I found some relevant songs for your request, "
                "but I couldn’t generate a full AI explanation right now."
            )

    return recommendation, results, error_message


def render_assistant_message(turn):
    recommendation_html = format_recommendation_html(turn.get("recommendation", ""))
    warning_html = ""

    if turn.get("error"):
        warning_html = (
            '<div class="soft-warning">'
            "AI explanation is unavailable right now. Showing the best search results instead."
            "</div>"
        )

    if not recommendation_html:
        recommendation_html = "I found some relevant songs for your request."

    render_html(
        '<div class="assistant-card">'
        '<div class="assistant-topline">'
        '<span class="assistant-dot"></span>'
        "<span>SongSense Recommendation</span>"
        "</div>"
        f"{warning_html}"
        '<div class="section-title">Recommended for your vibe</div>'
        f'<div class="recommendation-text">{recommendation_html}</div>'
        "</div>"
    )


def render_compact_header():
    render_html(
        '<div class="hero-wrap compact">'
        '<div class="main-title">🎵 SongSense AI</div>'
        '<div class="sub-text">Describe a feeling, memory, scene, or vibe — and let the bot find songs that fit.</div>'
        "</div>"
    )


def render_empty_state():
    render_html(
        '<div class="empty-state-shell">'
        '<div class="hero-wrap">'
        '<div class="main-title">🎵 SongSense AI</div>'
        '<div class="sub-text">Describe a feeling, memory, scene, or vibe — and let the bot find songs that fit.</div>'
        "</div>"
        '<div class="welcome-card">'
        '<div class="welcome-title">Your music companion is ready</div>'
        '<div class="welcome-text">'
        "Tell me what kind of song you want — a mood, a memory, a situation, or even a tiny dramatic sentence. "
        "I’ll try to find songs that match the feeling, not just the keywords."
        "</div>"
        '<div class="welcome-chip-row">'
        '<span class="welcome-chip">A sad song about missing someone</span>'
        '<span class="welcome-chip">Late-night rain vibes</span>'
        '<span class="welcome-chip">Something soft and romantic</span>'
        "</div>"
        "</div>"
        "</div>"
    )


def render_user_message(text: str):
    render_html(
        '<div class="user-row">'
        '<div class="chat-role-label">You</div>'
        f'<div class="user-card">{safe_text(text)}</div>'
        "</div>"
    )


def main():
    st.set_page_config(page_title="SongSense AI", page_icon="🎵", layout="wide")
    init_state()

    st.markdown(
        """
        <style>
            :root {
                --bg-1: #07152f;
                --bg-2: #081224;
                --border: rgba(255, 255, 255, 0.08);
                --text: #eef4ff;
                --muted: #9fb0d0;
                --muted-2: #c9d5ee;
                --accent: #8b5cf6;
                --accent-2: #ec4899;
                --shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
                --content-width: 920px;
            }

            html, body,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%) !important;
            }

            .stApp {
                background:
                    radial-gradient(circle at top, rgba(139,92,246,0.14), transparent 28%),
                    radial-gradient(circle at 85% 20%, rgba(236,72,153,0.10), transparent 22%),
                    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
                color: var(--text);
            }

            .block-container {
                max-width: var(--content-width);
                padding-top: 1rem;
                padding-bottom: 5rem;
            }

            header[data-testid="stHeader"] {
                background: transparent;
            }

            section[data-testid="stSidebar"] {
                display: none;
            }

            div[data-testid="stChatMessageAvatar"],
            div[data-testid="stChatMessageAvatarUser"],
            div[data-testid="stChatMessageAvatarAssistant"] {
                display: none !important;
            }

            div[data-testid="stChatMessage"] {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                gap: 0.35rem !important;
                padding-top: 0.12rem !important;
                padding-bottom: 0.12rem !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
            }

            div[data-testid="stChatMessageContent"] {
                width: 100% !important;
                background: transparent !important;
                padding: 0 !important;
            }

            .empty-state-shell {
                max-width: var(--content-width);
                margin: 6vh auto 0 auto;
            }

            .hero-wrap {
                text-align: center;
                margin-bottom: 1.1rem;
            }

            .hero-wrap.compact {
                margin-bottom: 0.75rem;
            }

            .main-title {
                font-size: 3rem;
                font-weight: 800;
                line-height: 1.08;
                margin-bottom: 0.38rem;
                letter-spacing: -0.02em;
                background: linear-gradient(90deg, #c084fc 0%, #f472b6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-wrap.compact .main-title {
                font-size: 2.25rem;
                margin-bottom: 0.16rem;
            }

            .sub-text {
                color: var(--muted);
                font-size: 1rem;
                line-height: 1.7;
            }

            .hero-wrap.compact .sub-text {
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .welcome-card {
                background: linear-gradient(180deg, rgba(18,27,47,0.92), rgba(13,20,36,0.92));
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1.2rem 1.2rem 1rem 1.2rem;
                box-shadow: var(--shadow);
                margin-bottom: 1.1rem;
            }

            .welcome-title {
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 0.65rem;
                color: #f8fbff;
            }

            .welcome-text {
                color: var(--muted-2);
                line-height: 1.78;
                font-size: 0.98rem;
            }

            .welcome-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.95rem;
            }

            .welcome-chip {
                background: rgba(139,92,246,0.12);
                border: 1px solid rgba(139,92,246,0.18);
                color: #ddd6fe;
                padding: 0.5rem 0.8rem;
                border-radius: 999px;
                font-size: 0.9rem;
                line-height: 1.15;
            }

            .user-row {
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                margin-bottom: 0.8rem;
            }

            .chat-role-label {
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #a9b7d6;
                margin-bottom: 0.22rem;
                margin-right: 0.02rem;
                text-align: right;
            }

            .user-card {
                width: fit-content;
                max-width: 80%;
                margin-left: auto !important;
                margin-right: 0 !important;
                background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(236,72,153,0.18));
                border: 1px solid rgba(236,72,153,0.16);
                border-radius: 18px;
                padding: 0.82rem 0.95rem;
                box-shadow: 0 10px 26px rgba(0,0,0,0.18);
                color: #f8fbff;
                line-height: 1.6;
            }

            .assistant-card {
                background: linear-gradient(180deg, rgba(18,28,48,0.96), rgba(14,21,37,0.96));
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 0.85rem;
            }

            .assistant-topline {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #cbd5e1;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                margin-bottom: 0.8rem;
                text-transform: uppercase;
            }

            .assistant-dot {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: linear-gradient(135deg, #8b5cf6, #ec4899);
                display: inline-block;
            }

            .section-title {
                font-size: 1.08rem;
                font-weight: 800;
                color: #ffffff;
                margin-bottom: 0.55rem;
            }

            .recommendation-text {
                color: var(--text);
                font-size: 1rem;
                line-height: 1.95;
            }

            .soft-warning {
                background: rgba(245, 158, 11, 0.10);
                border: 1px solid rgba(245, 158, 11, 0.18);
                color: #fde68a;
                border-radius: 14px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.9rem;
            }

            .empty-note {
                color: var(--muted);
            }

            div[data-testid="stBottomBlockContainer"],
            div[data-testid="stBottomBlockContainer"] > div,
            div[data-testid="stBottom"],
            div[data-testid="stBottom"] > div,
            footer {
                background: linear-gradient(
                    180deg,
                    rgba(7, 21, 47, 0) 0%,
                    rgba(8, 18, 36, 0.94) 42%,
                    rgba(8, 18, 36, 0.98) 100%
                ) !important;
                border-top: none !important;
                box-shadow: none !important;
            }

            /* ===== Chat input: centered, cleaner, no double box ===== */
            div[data-testid="stChatInput"] {
                background: transparent !important;
                border-top: none !important;
                padding: 0 0 0.58rem 0 !important;
                --primary-color: #6366f1 !important;
                --red-70: #6366f1 !important;
                --red-80: #6366f1 !important;
                --red-90: #818cf8 !important;
            }

            div[data-testid="stChatInput"] > div {
                width: min(760px, calc(100vw - 48px)) !important;
                max-width: none !important;
                margin: 0 auto !important;
                padding: 0 !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stChatInput"] form {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.55rem !important;
                background: rgba(15, 23, 42, 0.96) !important;
                border: 1px solid rgba(147, 197, 253, 0.76) !important;
                border-radius: 16px !important;
                box-shadow:
                    inset 0 0 0 1px rgba(147, 197, 253, 0.28),
                    0 8px 22px rgba(0, 0, 0, 0.14) !important;
                padding: 0.38rem 0.5rem !important;
                backdrop-filter: blur(8px);
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            div[data-testid="stChatInput"] form:hover {
                border: 1px solid rgba(165, 180, 252, 0.95) !important;
                box-shadow:
                    0 0 0 2px rgba(125, 140, 255, 0.28),
                    inset 0 0 0 1px rgba(191, 219, 254, 0.38),
                    0 10px 24px rgba(37, 99, 235, 0.22) !important;
            }

            div[data-testid="stChatInput"] form:focus-within {
                border: 1px solid rgba(191, 219, 254, 0.98) !important;
                box-shadow:
                    0 0 0 3px rgba(125, 140, 255, 0.36),
                    inset 0 0 0 1px rgba(224, 231, 255, 0.45),
                    0 12px 28px rgba(37, 99, 235, 0.28) !important;
            }

            /* Streamlit wrappers that may still render default red border */
            div[data-testid="stChatInput"] > div > div {
                background: transparent !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }

            div[data-testid="stChatInput"] [aria-invalid="true"],
            div[data-testid="stChatInput"] textarea:invalid,
            div[data-testid="stChatInput"] input:invalid {
                border-color: rgba(132, 154, 255, 0.86) !important;
                box-shadow: none !important;
                outline: none !important;
            }

            div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"],
            div[data-testid="stChatInput"] [data-baseweb="base-input"],
            div[data-testid="stChatInput"] [data-baseweb="base-input"] > div,
            div[data-testid="stChatInput"] [data-baseweb="textarea"],
            div[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                min-height: unset !important;
            }

            div[data-testid="stChatInput"] [data-baseweb="base-input"]::before,
            div[data-testid="stChatInput"] [data-baseweb="base-input"]::after,
            div[data-testid="stChatInput"] [data-baseweb="textarea"]::before,
            div[data-testid="stChatInput"] [data-baseweb="textarea"]::after {
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }

            div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {
                flex: 1 1 auto !important;
                width: 100% !important;
            }

            div[data-testid="stChatInput"] textarea,
            div[data-testid="stChatInput"] input {
                width: 100% !important;
                background: transparent !important;
                color: #eef4ff !important;
                -webkit-text-fill-color: #eef4ff !important;
                min-height: 30px !important;
                max-height: 120px !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                font-size: 1rem !important;
                line-height: 1.45 !important;
                padding: 0.18rem 0 !important;
                caret-color: #a78bfa !important;
            }

            div[data-testid="stChatInput"] textarea:focus,
            div[data-testid="stChatInput"] textarea:active,
            div[data-testid="stChatInput"] textarea:focus-visible,
            div[data-testid="stChatInput"] input:focus,
            div[data-testid="stChatInput"] input:active,
            div[data-testid="stChatInput"] input:focus-visible {
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
            }

            div[data-testid="stChatInputSubmitButton"] {
                flex: 0 0 auto !important;
            }

            div[data-testid="stChatInput"] button,
            div[data-testid="stChatInput"] button[kind],
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"] {
                width: 42px !important;
                height: 42px !important;
                min-width: 42px !important;
                position: relative !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border-radius: 12px !important;
                background: linear-gradient(135deg, #60a5fa 0%, #7c83ff 55%, #9b6bff 100%) !important;
                background-color: #6a72ff !important;
                border: none !important;
                color: #f8fbff !important;
                box-shadow:
                    0 0 0 1px rgba(224, 231, 255, 0.22),
                    inset 0 0 0 1px rgba(239, 246, 255, 0.30),
                    0 8px 18px rgba(79, 70, 229, 0.34) !important;
                transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease, background 0.16s ease !important;
            }

            /* Force arrow icon (hide Streamlit default play/stop glyphs) */
            div[data-testid="stChatInput"] button::before,
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"]::before {
                content: "↑";
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -54%);
                color: #ffffff;
                font-size: 20px;
                font-weight: 800;
                line-height: 1;
                pointer-events: none;
            }

            div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] > div {
                background: transparent !important;
                border: none !important;
            }

            div[data-testid="stChatInput"] button:hover,
            div[data-testid="stChatInput"] button[kind]:hover,
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"]:hover {
                background: linear-gradient(135deg, #7dd3fc 0%, #93c5fd 32%, #8b9cff 66%, #b38cff 100%) !important;
                background-color: #8791ff !important;
                box-shadow:
                    0 0 0 2px rgba(147, 197, 253, 0.28),
                    inset 0 0 0 1px rgba(255, 255, 255, 0.55),
                    0 12px 24px rgba(96, 165, 250, 0.40) !important;
                transform: translateY(-1px) scale(1.05) !important;
                filter: saturate(1.08) brightness(1.04) !important;
            }

            div[data-testid="stChatInput"] button:active,
            div[data-testid="stChatInput"] button[kind]:active,
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"]:active {
                transform: translateY(0) scale(0.98) !important;
                box-shadow:
                    inset 0 0 0 1px rgba(224, 231, 255, 0.30),
                    0 6px 14px rgba(37, 99, 235, 0.30) !important;
            }

            div[data-testid="stChatInput"] button:focus,
            div[data-testid="stChatInput"] button:focus-visible {
                outline: none !important;
                box-shadow:
                    0 0 0 2px rgba(125, 140, 255, 0.42),
                    inset 0 0 0 1px rgba(224, 231, 255, 0.34),
                    0 10px 22px rgba(59, 130, 246, 0.40) !important;
            }

            div[data-testid="stChatInput"] button svg,
            div[data-testid="stChatInput"] button path {
                fill: #ffffff !important;
                stroke: #ffffff !important;
            }

            div[data-testid="stChatInput"] button svg {
                width: 18px !important;
                height: 18px !important;
                display: block !important;
                margin: 0 auto !important;
            }

            div[data-testid="stChatInput"] button svg,
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"] svg {
                opacity: 0 !important;
            }

            div[data-testid="stChatInput"] button > div,
            div[data-testid="stChatInput"] button span,
            div[data-testid="stChatInput"] [data-testid^="stBaseButton"] > div {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                height: 100% !important;
            }

            @media (max-width: 768px) {
                .main-title {
                    font-size: 2.35rem;
                }

                .hero-wrap.compact .main-title {
                    font-size: 2rem;
                }

                .block-container {
                    padding-top: 0.8rem;
                    padding-bottom: 5rem;
                }

                .user-card {
                    max-width: 82%;
                }

                .empty-state-shell {
                    margin-top: 4vh;
                }

                div[data-testid="stChatInput"] > div {
                    width: min(700px, calc(100vw - 24px)) !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading models and vector index..."):
        df_chunks, index, model, reranker = load_system()

    if len(st.session_state.chat_history) == 0:
        render_empty_state()
    else:
        render_compact_header()

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            render_user_message(turn["query"])

        with st.chat_message("assistant"):
            render_assistant_message(turn)

    prompt = st.chat_input("Describe the song you want...")

    if prompt:
        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            st.stop()

        if cleaned_prompt.lower() in ["/clear", "/reset"]:
            st.session_state.chat_history = []
            st.rerun()

        with st.chat_message("user"):
            render_user_message(cleaned_prompt)

        with st.chat_message("assistant"):
            recommendation, results, error_message = run_chat_query(
                cleaned_prompt, df_chunks, index, model, reranker
            )

            temp_turn = {
                "query": cleaned_prompt,
                "recommendation": recommendation,
                "results": results,
                "error": error_message,
            }

            render_assistant_message(temp_turn)

        add_chat_turn(cleaned_prompt, recommendation, results, error_message)
        st.rerun()


if __name__ == "__main__":
    main()