from sentence_transformers import CrossEncoder

# Load once
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, documents):
    """
    Rerank documents using cross-encoder.
    Ensures all inputs are strings.
    """

    # Ensure we only extract document text
    cleaned_docs = []

    for doc in documents:
        if isinstance(doc, tuple):
            doc = doc[0]  # extract text if (text, score)

        if doc is not None:
            cleaned_docs.append(str(doc))

    if not cleaned_docs:
        return []

    # Create (query, doc) pairs
    pairs = [(str(query), doc) for doc in cleaned_docs]

    scores = reranker_model.predict(pairs)

    # Combine and sort
    ranked = list(zip(cleaned_docs, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked