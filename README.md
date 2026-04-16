# 🎵 SongSense AI

SongSense AI is an intelligent music recommendation system that helps you find the perfect song for any mood. By combining the speed of **FAISS Vector Search** with the reasoning capabilities of **Large Language Models (LLMs)**, SongSense AI understands the context of your request and suggests the most relevant songs along with personalized explanations.

![UI Overview](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Backend](https://img.shields.io/badge/Backend-Python-3776AB)
![Vector Search](https://img.shields.io/badge/Search-FAISS-0052CC)
![LLM](https://img.shields.io/badge/LLM-DeepSeek-blue)

---

## ✨ Features

- **Semantic Vector Search:** Uses `SentenceTransformers` to convert search queries into embeddings and performs ultra-fast similarity searches against hundreds of thousands of songs using `FAISS`.
- **Reranker:** Improves search precision by utilizing a Cross-Encoder to re-score the retrieval results based on deep semantic context.
- **LLM-Powered Recommendations:** Leverages DeepSeek (via OpenRouter) to read the top semantic matches and explain *why* they fit the user's specific request in a friendly, conversational tone.
- **Modern UI:** A clean, responsive interface built entirely in standard Python using `Streamlit`.

## 📁 Project Structure

```text
project/
├── app.py                   # Main Streamlit UI interface
├── config.py                # Configuration and common file paths
├── engines/
│   ├── engine_search.py     # Logic for semantic vector search natively 
│   └── engine_llm.py        # Logic for querying OpenRouter Deepseek model
├── pipeline/
│   └── pipeline_build.py    # Raw dataset downloading and Faiss index building
├── data/                    # Generated vector mappings and databases
│   ├── df_chunks.csv        # Preprocessed chunks data
│   ├── faiss.index          # FAISS vector database
│   └── embeddings.npy       # NumPy embeddings array
└── README.md                # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install streamlit pandas numpy faiss-cpu sentence-transformers openai kagglehub
```

### 2. Configure API Keys

In `engines/engine_llm.py`, ensure that you add your OpenRouter API key so the LLM can generate responses:

```python
# engines/engine_llm.py
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)
```

### 3. Build the Vector Index (Run Once)

Before starting the web app, you must download the dataset and build the vector database. This will create the `faiss.index`, `df_chunks.csv`, and `embeddings.npy` inside your `data/` folder.

*(Note: The embedding process can take a while depending on your CPU/GPU.)*

```bash
python pipeline/pipeline_build.py
```

### 4. Run the Web App

Once the data is preprocessed and the index is built, you can run the Streamlit application:

```bash
streamlit run app.py
```

The app will load in your default web browser at `http://localhost:8501`.

---

## 🛠️ Built With
* [Streamlit](https://streamlit.io/) - The web framework used.
* [FAISS](https://faiss.ai/) - Library for efficient similarity search and clustering of dense vectors.
* [SentenceTransformers](https://sbert.net/) - Framework for state-of-the-art sentence, text and image embeddings.
* [OpenRouter (DeepSeek)](https://openrouter.ai/) - LLM provider for conversational AI.

## 📝 License
This project is for educational purposes. Data is sourced from the [900k-spotify dataset on Kaggle](https://www.kaggle.com/datasets/devdope/900k-spotify).

## Must do 
pip install streamlit pandas numpy faiss-cpu sentence-transformers openai kagglehub
python pipeline/pipeline_build.py
streamlit run app.py
## OR
py -3.11 -m pip install sentence-transformers faiss-cpu streamlit openai
py -3.11 -m pipeline/pipeline_build.py
py -3.11 -m streamlit run app.py