import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = client.get_or_create_collection("documents")


def add_chunks(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)

    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )


def query_vector(query, k=5):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]  # smaller = better

    # Convert distance → similarity
    scores = [1 - d for d in distances]

    return list(zip(documents, scores))