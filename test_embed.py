from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the multilingual embedding model (downloads once, then cached)
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

if __name__ == "__main__":
    with open("cleaned_output.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension per chunk: {embeddings[0].shape}")