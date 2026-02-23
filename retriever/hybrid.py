def normalize(scores):
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s == 0:
        return [1 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_rank(vector_results, bm25_results, alpha=0.6, k=5):
    combined = {}

    # Add vector results
    for doc, score in vector_results:
        combined[doc] = {"vector": score, "bm25": 0}

    # Add BM25 results
    for doc, score in bm25_results:
        if doc not in combined:
            combined[doc] = {"vector": 0, "bm25": score}
        else:
            combined[doc]["bm25"] = score

    # Normalize scores
    vector_scores = normalize([v["vector"] for v in combined.values()])
    bm25_scores = normalize([v["bm25"] for v in combined.values()])

    docs = list(combined.keys())

    final_scores = []
    for i in range(len(docs)):
        score = alpha * vector_scores[i] + (1 - alpha) * bm25_scores[i]
        final_scores.append((docs[i], score))

    # Sort by final score
    final_scores.sort(key=lambda x: x[1], reverse=True)

    return final_scores[:k]
