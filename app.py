import streamlit as st
import sys
import os

# Add the project root strictly to sys.path so it can find config, engines
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.engine_search import load_system, search
from engines.engine_llm import recommend_songs

def main():
    st.set_page_config(page_title="SongSense AI", page_icon="🎵", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 3.5rem;
            background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.5rem;
            margin-top: -2rem;
        }
        .sub-text {
            text-align: center;
            color: #888888;
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
        .recommendation-box {
            background: rgba(78, 205, 196, 0.1);
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #4ECDC4;
            margin-top: 1rem;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .song-card {
            background: rgba(255, 107, 107, 0.05);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 107, 107, 0.2);
            transition: transform 0.2s ease;
        }
        .song-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 107, 107, 0.1);
        }
        .song-title {
            color: #FF6B6B;
            font-weight: bold;
            font-size: 1.2rem;
        }
        .badge {
            background: rgba(78, 205, 196, 0.2);
            color: #4ECDC4;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🎵 SongSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Find the perfect song for any mood, powered by Vector Search & LLMs.</div>', unsafe_allow_html=True)

    with st.spinner("Loading models and vector index (this may take a moment on first load)..."):
        df_chunks, index, model, reranker = load_system()
        
    query = st.text_input("What kind of song are you looking for?", placeholder="E.g., A sad song about missing someone on Christmas")
    
    if st.button("Search & Recommend", type="primary"):
        if query.strip():
            with st.spinner("Searching for the best semantic matches..."):
                results = search(query, df_chunks, index, model, reranker)
            
            st.divider()
            
            col1, col2 = st.columns([1.2, 0.8])
            
            # AI Recommendation column
            with col1:
                st.subheader("✨ AI Music Recommendation")
                with st.spinner("Analyzing songs and generating recommendation via LLM..."):
                    try:
                        recommendation = recommend_songs(query, results)
                        st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error communicating with LLM: {str(e)}")
            
            # Best Matches column
            with col2:
                st.subheader("🎯 Top Context Matches")
                for i, row in results.reset_index(drop=True).iterrows():
                    genre = str(row.get('genre', 'Unknown')).replace('[', '').replace(']', '').replace("'", "")
                    emotion = str(row.get('emotion', 'Unknown'))
                    st.markdown(f"""
                        <div class="song-card">
                            <div class="song-title">{i+1}. {row['song']}</div>
                            <div style="color: #888; margin-bottom: 0.8rem;">by {row['artist']}</div>
                            <div>
                                <span class="badge">{genre}</span>
                                <span class="badge">{emotion}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a query first!")

if __name__ == "__main__":
    main()
