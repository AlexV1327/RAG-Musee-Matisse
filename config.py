import os, hashlib
from pathlib import Path

BASE_URL = "https://www.musee-matisse-nice.org"
START_PATHS = ["/fr/"]  # cible la version française (slash final)
DATA_DIR = Path("./matisse_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers de données
PAGES_FILE  = DATA_DIR / "pages.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Crawl
USER_AGENT = "RAG-educatif/1.0"
REQUEST_TIMEOUT = 20
THROTTLE_SECONDS = 0.8
MAX_PAGES = 5  # POC

# Embeddings (une seule fois)
EMB_MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64

# LLM
API_BASE  = os.getenv("API_BASE", "https://api.mistral.ai")
API_KEY   = os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY")
API_MODEL = os.getenv("MODEL", "mistral-small-latest")
OLLAMA_MODEL = "mistral:7b-instruct"

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8","ignore")).hexdigest()[:16]

TARGET_URLS = [
    "https://www.musee-matisse-nice.org/fr/collection/historique-de-la-collection",
    "https://www.musee-matisse-nice.org/fr/informations-pratiques",
    "https://www.musee-matisse-nice.org/fr/collection/bibliotheque-de-matisse",
    "https://www.musee-matisse-nice.org/fr/musee/histoire-du-musee",
    "https://www.musee-matisse-nice.org/fr/exposition/matisse-mediterranees",
]

# Index vectoriel
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "chunk_meta.jsonl"

# Recherche
TOP_K = 5

# RAG (taille contexte LLM)
MAX_CONTEXT_CHARS = 2200
