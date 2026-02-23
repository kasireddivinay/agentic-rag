from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------------------
# Initialize Groq Client
# ---------------------------------------
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------------
# 1️⃣ Answer Generator
# ---------------------------------------
def generate_answer(query, docs):
    """
    Generate grounded answer using retrieved documents.
    """

    context = "\n\n".join(docs)

    prompt = f"""
You are a factual assistant.

Use ONLY the context below to answer the question.
Do NOT use outside knowledge.

If the answer is not present in the context, reply exactly:
"Not found in context."

---------------------
Context:
{context}
---------------------

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------
# 2️⃣ Smart Query Rewriter
# ---------------------------------------
def rewrite_query(query):
    """
    Rewrite user query to improve retrieval quality.
    Returns keyword-optimized version.
    """

    prompt = f"""
Rewrite the following question to improve document retrieval.

Focus on:
- Important keywords
- Core concepts
- Search-friendly phrasing

Return ONLY the rewritten query.

Original Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()