# text_utils.py

def chunk_text(text, chunk_size=500):
    """
    Splits text into smaller chunks for embeddings and Pinecone storage.
    Default chunk size = 500 characters.
    """
    chunks = []
    text = text.strip()
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks
