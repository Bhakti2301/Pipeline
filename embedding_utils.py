import os
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: list[str]):
    """
    Create embeddings for a list of strings using OpenAI API.
    """
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
