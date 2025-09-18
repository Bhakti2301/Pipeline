def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        slice_ = text[i:end].strip()
        if slice_:
            chunks.append(slice_)
        i += chunk_size - overlap
    return chunks
