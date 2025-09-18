# from supabase_utils import insert_document_record  # ✅ remove insert_chunks import
# from pinecone_utils import upsert_chunks
# from text_utils import chunk_text

# def store_plain_text(text, user_id="00000000-0000-0000-0000-000000000000"):
#     # 1) Insert into Supabase
#     file_name = f"agent_input.txt"
#     doc = insert_document_record(user_id, file_name, text)

#     # 2) Chunk the text
#     chunks = chunk_text(text)

#     # 3) Upsert into Pinecone
#     upsert_chunks(doc['id'], chunks)

#     print(f"✅ Stored {len(chunks)} chunks in Supabase & Pinecone for document {doc['id']}")





import os
from supabase_utils import insert_document_record
from pinecone_utils import upsert_chunks, get_embedding
from uuid import UUID
from typing import List

# Split text into chunks
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Insert chunks into Supabase
# def insert_chunks_to_supabase(document_id: UUID, chunks: List[str], embeddings: List[List[float]]):
#     """Insert text chunks with embeddings into Supabase doc_chunks table."""
#     from supabase_utils import supabase
#     for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
#         res = supabase.table("doc_chunks").insert({
#             "document_id": str(document_id),
#             "chunk_index": i,
#             "content": chunk,
#             "embedding": emb
#         }).execute()
#         if res.errors:
#             print(f"❌ Failed to insert chunk {i}: {res.errors}")
#     print(f"✅ Inserted {len(chunks)} chunks into Supabase.")


def insert_chunks_to_supabase(document_id, chunks, embeddings):
    """Insert text chunks with embeddings into Supabase doc_chunks table."""
    from supabase_utils import supabase

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
       res = supabase.table("doc_chunks").insert({
           "document_id": str(document_id),
           "chunk_index": i,
           "content": chunk,
           "embedding": emb
       }).execute()

       
       if not res.data or len(res.data) == 0:
           print(f"❌ Failed to insert chunk {i}. Response: {res}")
       else:
           print(f"✅ Inserted chunk {i} successfully: {res.data[0]['id']}")

# Full store function
def store_plain_text(text: str, user_id: str, file_name: str = "document.txt"):
    # 1️⃣ Store metadata in Supabase
    doc = insert_document_record(user_id=user_id, file_name=file_name, text=text)
    document_id = doc["id"]
    print(f"✅ Stored document metadata in Supabase: {document_id}")

    # 2️⃣ Chunk the text
    chunks = chunk_text(text)

    # 3️⃣ Generate embeddings for each chunk
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # 4️⃣ Store chunks in Supabase
    insert_chunks_to_supabase(document_id, chunks, embeddings)

    # 5️⃣ Upsert chunks into Pinecone
    upsert_chunks(chunks, embeddings, document_id=document_id)
