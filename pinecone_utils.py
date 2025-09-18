# import os
# from pinecone import Pinecone
# from openai import OpenAI

# # Create Pinecone client instance
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# INDEX_NAME = "docsearch"  # Change to your actual index name

# # Initialize OpenAI client
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_embedding(text):
#     """Generate an embedding vector for a given text chunk."""
#     response = openai_client.embeddings.create(
#         model="text-embedding-3-small",  # ✅ Make sure it matches your index dimension (1536)
#         input=text
#     )
#     return response.data[0].embedding





# def upsert_chunks(doc_id, chunks):
#     """Upsert text chunks with embeddings into Pinecone."""
#     index = pc.Index(INDEX_NAME)
#     vectors = []

#     for i, chunk in enumerate(chunks):
#         embedding = get_embedding(chunk)
#         vectors.append({
#             "id": f"{doc_id}-chunk-{i}",  # unique per document
#             "values": embedding,
#             "metadata": {"text": chunk}
#         })

#     index.upsert(vectors=vectors)
#     print(f"✅ Upserted {len(vectors)} chunks to Pinecone.")














import os
from pinecone import Pinecone
from openai import OpenAI

# Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "docsearch"

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    """Generate embedding vector for a text chunk."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def upsert_chunks(chunks, embeddings, document_id: str):
    """Upsert chunks into Pinecone with document_id as prefix for unique IDs."""
    index = pc.Index(INDEX_NAME)
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{document_id}-chunk-{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors)
    print(f"✅ Upserted {len(vectors)} chunks to Pinecone.")



