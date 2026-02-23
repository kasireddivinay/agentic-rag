from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")


def faithfulness_score(answer, context_docs):
    """
    Compute maximum similarity between answer and individual chunks.
    """

    answer_embedding = model.encode([answer])

    max_score = -1

    for doc in context_docs:
        chunk_embedding = model.encode([doc])
        score = cosine_similarity(answer_embedding, chunk_embedding)[0][0]

        if score > max_score:
            max_score = score

    return float(max_score)


def confidence_score(faithfulness):
    return round(faithfulness * 100, 2)