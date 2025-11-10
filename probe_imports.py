import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"]="1"
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

def try_import(name, stmt):
    print(f"--> importing {name} ...", flush=True)
    exec(stmt, globals(), globals())
    print(f"OK: {name}")

try_import("faiss", "import faiss")
try_import("torch", "import torch")
try_import("sentence_transformers", "from sentence_transformers import SentenceTransformer")
try_import("chromadb", "import chromadb")
print("ALL GOOD")
