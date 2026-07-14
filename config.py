import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
IMAGES_DIR   = os.path.join(DATA_DIR, "images")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for d in [DATA_DIR, IMAGES_DIR, CHROMA_DB_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Model Names ────────────────────────────────────────────────────────────
CLIP_MODEL_NAME  = "hf-hub:Marqo/marqo-fashionCLIP"
GEMINI_MODEL     = "gemini-2.0-flash"
GROQ_TEXT_MODEL  = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
