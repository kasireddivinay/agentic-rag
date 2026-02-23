from retriever.vector_store import query_vector
from retriever.bm25 import query_bm25
from retriever.hybrid import hybrid_rank
from retriever.reranker import rerank
from generator.llm_generator import generate_answer, rewrite_query
from evaluation.metrics import faithfulness_score


def autonomous_reasoning(query, threshold=0.7, max_attempts=3):

    attempts_data = []
    current_query = query
    best_answer = None
    best_score = -1

    for attempt in range(max_attempts):

        # 1️⃣ Retrieve
        vector_results = query_vector(current_query)
        bm25_results = query_bm25(current_query)

        hybrid_results = hybrid_rank(vector_results, bm25_results)
        reranked = rerank(current_query, hybrid_results)

        top_docs = [doc for doc, score in reranked[:5]]

        # 2️⃣ Generate
        answer = generate_answer(current_query, top_docs)

        # 3️⃣ Evaluate
        score = faithfulness_score(answer, top_docs)

        # Store attempt trace
        attempts_data.append({
            "query": current_query,
            "faithfulness": score,
            "chunks": top_docs
        })

        # Track best
        if score > best_score:
            best_score = score
            best_answer = answer

        # Accept if good enough
        if score >= threshold:
            return answer, score, attempts_data

        # Otherwise rewrite
        current_query = rewrite_query(current_query)

    # If no attempt passed threshold
    return best_answer, best_score, attempts_data