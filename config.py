import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing in the .env file.")
    
    DATA_DIR=os.path.join(os.path.dirname(__file__),"data")

    EMBEDDING_MODEL="text-embedding-3-small"

    CHUNK_BREAKPOINT_TYPE = "percentile"
    CHUNK_BREAKPOINT_THRESHOLD = 80