import os
import io
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from pathlib import Path

from supabase import create_client, Client
from openai import OpenAI

from pdfminer.high_level import extract_text as pdf_extract_text
import mammoth

# Load .env from this file's directory
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

PORT = int(os.getenv("PORT", "4100"))
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")
KEEP_ORIGINAL = str(os.getenv("KEEP_ORIGINAL", "false")).lower() == "true"
STORE_AS_TEXT_ONLY = str(os.getenv("STORE_AS_TEXT_ONLY", "true")).lower() == "true"
DEBUG = str(os.getenv("DEBUG", "true")).lower() == "true"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE (ensure python_server_clean/.env exists and has both)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return PlainTextResponse("\n".join([
        "Doc Pipeline API (Python Clean) is running.",
        "",
        "Available endpoints:",
        "GET  /health",
        "GET  /history?user_id=<id>",
        "GET  /search?q=<query>&topK=5[&documentId=<id>]",
        "POST /upload  (multipart/form-data: file, user_id)",
        "GET  /documents/{id}/text",
        "",
        f"Config: KEEP_ORIGINAL={KEEP_ORIGINAL}, STORE_AS_TEXT_ONLY={STORE_AS_TEXT_ONLY}, DEBUG={DEBUG}",
    ]))

@app.get("/health")
async def health():
    return {"ok": True}

async def store_file_to_supabase(data: bytes, file_name: str, mime: str) -> str:
    path = f"{uuid.uuid4()}/{file_name}"
    res = supabase.storage.from_(SUPABASE_BUCKET).upload(path, data, {
        "contentType": mime,
        "upsert": False,
    })
    if res.get("error"):
        raise HTTPException(status_code=500, detail=str(res["error"]))
    return path

async def insert_document_record(user_id: Optional[str], file_name: str, storage_path: Optional[str], mime_type: str, size_bytes: int):
    r = supabase.table("documents").insert({
        "user_id": user_id,
        "file_name": file_name,
        "storage_path": storage_path,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
    }).execute()
    if r.error:
        raise HTTPException(status_code=500, detail=str(r.error))
    data = r.data or []
    return data[0] if data else None

async def embed_texts(texts: List[str]):
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    res = openai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]

async def extract_text(file_bytes: bytes, file_name: str, mime: str) -> str:
    name_lower = file_name.lower()
    if "pdf" in mime or name_lower.endswith(".pdf"):
        with io.BytesIO(file_bytes) as f:
            return pdf_extract_text(f)
    if "word" in mime or name_lower.endswith(".docx"):
        with io.BytesIO(file_bytes) as f:
            r = mammoth.extract_raw_text(f)
            return r.value or ""
    # fallback: treat as text
    return file_bytes.decode("utf-8", errors="replace")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        slice_ = text[i:end].strip()
        if slice_:
            chunks.append(slice_)
        i += chunk_size - overlap
    return chunks

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    try:
        # normalize user_id: accept UUID string or set None
        def normalize_user_id(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            try:
                return str(uuid.UUID(val))
            except Exception:
                return None

        user_id_norm = normalize_user_id(user_id)

        data = await file.read()

        # extract text
        full_text_raw = await extract_text(data, file.filename, file.content_type or "application/octet-stream")
        # Coerce to string defensively
        if isinstance(full_text_raw, str):
            full_text = full_text_raw
        elif isinstance(full_text_raw, (bytes, bytearray)):
            full_text = full_text_raw.decode("utf-8", errors="replace")
        else:
            full_text = str(full_text_raw or "")

        # determine storage strategy
        storage_path = None
        stored_file_name = file.filename
        stored_mime = file.content_type or "application/octet-stream"
        stored_size = len(data)

        base_name = os.path.splitext(file.filename)[0]

        if not isinstance(full_text, str):
            print("DEBUG: full_text is not a string but", type(full_text))
            full_text = str(full_text or "")

        
        if KEEP_ORIGINAL:
            # store original file
            storage_path = await store_file_to_supabase(data, file.filename, stored_mime)
        elif STORE_AS_TEXT_ONLY:
            # store only extracted text as .txt
            try:
                txt_bytes = full_text.encode("utf-8")
            except Exception as e:
                print(f"DEBUG: Error encoding full_text: {e}")
                txt_bytes = b""

            stored_file_name = f"{base_name}.txt"
            stored_mime = "text/plain"
            stored_size = len(txt_bytes)
            storage_path = await store_file_to_supabase(txt_bytes, stored_file_name, stored_mime)
        # else: no storage (requires DB column to allow NULL)
        print(f"DEBUG: stored_file_name={stored_file_name}, stored_mime={stored_mime}, stored_size={stored_size}, storage_path={storage_path}")
        print(f"DEBUG: type(full_text)={type(full_text)}, length={len(full_text) if isinstance(full_text, str) else 'N/A'}")


        # insert document row
        doc = await insert_document_record(
            user_id=user_id_norm,
            file_name=stored_file_name if (KEEP_ORIGINAL or STORE_AS_TEXT_ONLY) else base_name,
            storage_path=storage_path,
            mime_type=stored_mime if (KEEP_ORIGINAL or STORE_AS_TEXT_ONLY) else "text/plain",
            size_bytes=stored_size if (KEEP_ORIGINAL or STORE_AS_TEXT_ONLY) else len(full_text.encode("utf-8")),
        )

        # chunk + embed
        chunks = chunk_text(full_text)
        if chunks:
            vectors = await embed_texts(chunks)
            rows = [{
                "document_id": doc["id"],
                "chunk_index": i,
                "content": chunks[i],
                "embedding": vectors[i],
            } for i in range(len(chunks))]
            ins = supabase.table("doc_chunks").insert(rows).execute()
            if ins.error:
                raise HTTPException(status_code=500, detail=str(ins.error))

        return {"document": doc}
    except HTTPException:
        raise
    except Exception as e:
    # surface error message when DEBUG is enabled
        if DEBUG:
            print("DEBUG: Upload error", repr(e), type(e))
            print("DEBUG: Response to PlainTextResponse will be:", f"Upload failed: {type(e).__name__}: {e}", type(f"Upload failed: {type(e).__name__}: {e}"))
            return PlainTextResponse(f"Upload failed: {type(e).__name__}: {e}", status_code=500)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.get("/history")
async def history(user_id: Optional[str] = None):
    q = supabase.table("documents").select("*").order("uploaded_at", desc=True)
    if user_id:
        q = q.eq("user_id", user_id)
    r = q.execute()
    if r.error:
        raise HTTPException(status_code=500, detail=str(r.error))

    docs = r.data
    # add signed urls if storage_path exists
    out = []
    for d in docs:
        if d.get("storage_path"):
            signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(d["storage_path"], 60 * 60)
            if signed.get("error"):
                raise HTTPException(status_code=500, detail=str(signed["error"]))
            d = {**d, "file_url": signed["signedURL"]}
        out.append(d)
    return {"documents": out}

@app.get("/search")
async def search(q: str, topK: int = 5, documentId: Optional[str] = None):
    [emb] = await embed_texts([q])
    rpc = supabase.rpc("match_chunks", {
        "query_embedding": emb,
        "match_count": topK,
        "p_document_id": documentId,
    }).execute()
    if rpc.error:
        raise HTTPException(status_code=500, detail=str(rpc.error))
    return {"results": rpc.data}

@app.get("/documents/{doc_id}/text", response_class=PlainTextResponse)
async def document_text(doc_id: str):
    r = supabase.table("doc_chunks").select("content, chunk_index").eq("document_id", doc_id).order("chunk_index", asc=True).execute()
    if r.error:
        raise HTTPException(status_code=500, detail=str(r.error))
    text = "\n\n".join([row.get("content") or "" for row in r.data])
    return text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
