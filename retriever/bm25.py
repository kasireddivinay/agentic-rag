from rank_bm25 import BM25Okapi

bm25 = None
corpus = []
raw_docs = []

def initialize_bm25(chunks):
    global bm25, corpus, raw_docs

    raw_docs = chunks
    corpus = [doc.split(" ") for doc in chunks]
    bm25 = BM25Okapi(corpus)


def query_bm25(query, k=5):
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [(raw_docs[i], scores[i]) for i in ranked_indices]