import os
import datetime
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)

def insert_document_record(user_id, file_name, text):
    """Insert a document row into Supabase table."""
    now = datetime.datetime.utcnow().isoformat()
    
    # ✅ Generate a fake storage path since we are not actually uploading a file
    fake_storage_path = f"agent_inputs/{file_name}"

    data = {
        "user_id": user_id,
        "file_name": file_name,
        "storage_path": fake_storage_path,  # ✅ Required for NOT NULL
        "mime_type": "text/plain",
        "size_bytes": len(text.encode("utf-8")),
        "uploaded_at": now
    }

    result = supabase.table("documents").insert(data).execute()
    return result.data[0] if result.data else None
