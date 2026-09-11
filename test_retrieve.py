import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()  

DB_PASSWORD = os.getenv("DB_PASSWORD")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="document_query_db",
        user="postgres",
        password=DB_PASSWORD
    )

def retrieve_chunks(query, top_k=3):
    # Step 1: Embed the query using the same model
    query_embedding = model.encode(query).tolist()

    # Step 2: Search Postgres for the closest chunks
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, source_file, chunk_text, embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s
        """,
        (query_embedding, top_k)
    )

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results

if __name__ == "__main__":
    query = "What is Flutter used for?"
    results = retrieve_chunks(query)

    print(f"Query: {query}\n")
    for i, (chunk_id, source_file, chunk_text, distance) in enumerate(results):
        print(f"--- Result {i+1} (distance: {distance:.4f}) ---")
        print(f"Source: {source_file}")
        print(f"Text: {chunk_text[:150]}...")
        print()