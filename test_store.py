import psycopg2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv() 

DB_PASSWORD = os.getenv("DB_PASSWORD")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", " ", ""]
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    return model.encode(chunks)

# --- Database connection ---
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="document_query_db",
        user="postgres",
        password=DB_PASSWORD
    )

def store_chunks(source_file, chunks, embeddings):
    conn = get_connection()
    cur = conn.cursor()

    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO document_chunks (source_file, chunk_text, embedding)
            VALUES (%s, %s, %s)
            """,
            (source_file, chunk, embedding.tolist())
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Stored {len(chunks)} chunks into the database.")

if __name__ == "__main__":
    filename = "sample.pdf"
    
    with open("cleaned_output.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    store_chunks(filename, chunks, embeddings)