🚀 Autonomous Self-Correcting RAG Agent

Production-ready Agentic RAG system with hybrid retrieval, cross-encoder reranking, query rewriting, and faithfulness evaluation deployed using Streamlit Cloud.

🌐 Live Demo

🔗 https://agentic-rag-project.streamlit.app

🧠 Project Overview

This project implements an Autonomous Self-Correcting Retrieval-Augmented Generation (RAG) Agent designed to:
* Retrieve relevant context using hybrid search (BM25 + Vector Search)
* Rerank results using Cross-Encoder models
* Rewrite queries when confidence is low
* Evaluate faithfulness of generated responses
* Provide confidence and latency metrics
* Deploy as a live interactive dashboard


🏗 Architecture

User Query
    ↓
Hybrid Retrieval (BM25 + Vector DB)
    ↓
Cross-Encoder Reranking
    ↓
LLM Generation (Groq)
    ↓
Faithfulness Evaluation
    ↓
Self-Correction Loop (if needed)
    ↓
Final Answer + Metrics



⚙️ Tech Stack

Python
Streamlit
Groq LLM API
Sentence Transformers
ChromaDB
Rank-BM25
Scikit-learn
NumPy



📊 Features

Hybrid search improves retrieval accuracy
Cross-encoder reranking boosts relevance
Confidence scoring for reliability
Smart query rewriting loop
Live performance metrics (Faithfulness, Confidence, Latency)
Public cloud deployment


📦 Installation (Local)
git clone https://github.com/kasireddivinay/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
streamlit run app.py