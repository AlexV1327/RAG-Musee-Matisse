import os, hashlib
from pathlib import Path

# CRAWL / DONNÉES
BASE_URL = "https://www.musee-matisse-nice.org"
DATA_DIR = Path("./matisse_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers de données
PAGES_FILE  = DATA_DIR / "pages.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Crawl
USER_AGENT = "RAG-educatif/1.0 (+https://example.local)"
REQUEST_TIMEOUT = 20
THROTTLE_SECONDS = 0.8

# URLs ciblées (liste blanche) – sécurise le crawl
TARGET_URLS = [
    "https://www.musee-matisse-nice.org/fr/",
    "https://www.musee-matisse-nice.org/fr/collections",
    "https://www.musee-matisse-nice.org/fr/collection/bibliotheque-de-matisse",
    "https://www.musee-matisse-nice.org/fr/musee/histoire-du-musee",
    "https://www.musee-matisse-nice.org/fr/exposition/matisse-mediterranees",
]

# INDEX VECTORIEL
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "chunk_meta.jsonl"

# RECHERCHE / CHUNKING
TOP_K = 5
TARGET_CHARS   = 900   # taille visée d’un chunk
MIN_CHARS      = 350   # taille minimale avant de couper
OVERLAP_CHARS  = 120   # chevauchement entre chunks
MAX_CONTEXT_CHARS = 8_000  # limite de contexte envoyé au LLM

# EMBEDDINGS
EMB_MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64

# LLM (Mistral)
API_MODEL = os.getenv("API_MODEL", "mistral-large-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# HELPERS
sha16 = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
