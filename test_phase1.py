from ingestion.loader import load_documents
from ingestion.chunker import chunk_text

from retriever.vector_store import add_chunks
from retriever.bm25 import initialize_bm25

from agent.reasoning_loop import autonomous_reasoning


# -------------------------------
# 1. Load Documents
# -------------------------------
docs = load_documents("data")

# -------------------------------
# 2. Chunk Documents
# -------------------------------
all_chunks = []
for doc in docs:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

# -------------------------------
# 3. Add to Vector DB
# -------------------------------
add_chunks(all_chunks)

# -------------------------------
# 4. Initialize BM25
# -------------------------------
initialize_bm25(all_chunks)

# -------------------------------
# 5. Ask Question
# -------------------------------
query = input("\nEnter your question: ")

# -------------------------------
# 6. Run Autonomous Agent Loop
# -------------------------------
final_answer, final_score = autonomous_reasoning(query)

# -------------------------------
# 7. Final Output
# -------------------------------
print("\n==============================")
print("🧠 FINAL ANSWER")
print("==============================\n")

print(final_answer)

print("\n------------------------------")
print("📊 FINAL FAITHFULNESS SCORE:", round(final_score, 3))
print("------------------------------")

if final_score < 0.7:
    print("⚠ Low confidence. Agent attempted improvement.")
else:
    print("✅ High confidence. Answer grounded.")