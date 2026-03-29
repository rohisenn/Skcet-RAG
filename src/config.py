import os
from dotenv import load_dotenv


load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_PERSIST_DIR = "chroma_db"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "skcet_admin")
