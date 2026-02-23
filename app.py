import streamlit as st
import time
from agent.reasoning_loop import autonomous_reasoning
from ingestion.loader import load_documents
from ingestion.chunker import chunk_text
from retriever.vector_store import add_chunks
from retriever.bm25 import initialize_bm25

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Autonomous RAG Agent",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.big-title {
    font-size: 38px;
    font-weight: 700;
    color: white;
}
.subtitle {
    font-size: 16px;
    color: #94a3b8;
}
.answer-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 16px;
}
.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.success {
    color: #22c55e;
    font-weight: bold;
}
.warning {
    color: #facc15;
    font-weight: bold;
}
.danger {
    color: #ef4444;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Initialize System Once
# -------------------------
@st.cache_resource
def initialize_system():
    docs = load_documents("data")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    add_chunks(all_chunks)
    initialize_bm25(all_chunks)

initialize_system()

# -------------------------
# Header
# -------------------------
st.markdown('<div class="big-title">🧠 Autonomous Self-Correcting RAG Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid Retrieval + Reranking + Evaluation + Smart Reflection</div>', unsafe_allow_html=True)
st.write("")

# -------------------------
# Input
# -------------------------
query = st.text_input("Enter your question:")

if st.button("🚀 Run Agent") and query:

    start_time = time.time()

    with st.spinner("Thinking..."):
        final_answer, final_score, attempts = autonomous_reasoning(query)

    latency = round(time.time() - start_time, 2)

    # -------------------------
    # Confidence Label
    # -------------------------
    if final_score >= 0.75:
        confidence_label = "High Confidence"
        css_class = "success"
    elif final_score >= 0.6:
        confidence_label = "Moderate Confidence"
        css_class = "warning"
    else:
        confidence_label = "Low Confidence"
        css_class = "danger"

    st.success("Agent Completed ✅")

    col1, col2 = st.columns([2, 1])

    # -------------------------
    # Left Column (Answer + Chunks)
    # -------------------------
    with col1:
        st.markdown("### 📌 Final Answer")
        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

        st.markdown("### 🔎 Retrieved Chunks & Attempts")

        for i, attempt in enumerate(attempts):
            with st.expander(f"Attempt {i+1} | Faithfulness: {round(attempt['faithfulness'],3)}"):
                st.write("**Query Used:**", attempt["query"])
                for chunk in attempt["chunks"]:
                    st.markdown(f"- {chunk}")

    # -------------------------
    # Right Column (Metrics)
    # -------------------------
    with col2:
        st.markdown("### 📊 Metrics")

        st.markdown(f"""
        <div class="metric-card">
            <h2>{round(final_score,3)}</h2>
            <p>Faithfulness Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h2>{round(final_score*100,2)}%</h2>
            <p>Confidence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h2>{latency}s</h2>
            <p>Latency</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p class='{css_class}'>{confidence_label}</p>", unsafe_allow_html=True)