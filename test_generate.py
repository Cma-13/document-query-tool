import ollama

def generate_answer(query, retrieved_chunks):
    # Build the context from retrieved chunks
    context = "\n\n".join([chunk_text for (_, _, chunk_text, _) in retrieved_chunks])

    # Build the prompt using proper prompt engineering structure
    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
If the answer isn't in the context, say you don't know - do not make up information.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.generate(model='llama3.2:1b', prompt=prompt)
    return response['response']

if __name__ == "__main__":
    from test_retrieve import retrieve_chunks

    query = "What is your name?"
    results = retrieve_chunks(query)

    answer = generate_answer(query, results)

    print(f"Query: {query}\n")
    print(f"Answer: {answer}")