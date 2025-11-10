RAG Musée Matisse 

Une petite appli RAG (Retrieval-Augmented Generation) avec :
 - FAISS pour la recherche vectorielle
 - Sentence-Transformers pour les embeddings
 - Streamlit pour l’interface
 - Mistral pour la génération de texte

# Prérequis :

 - Python 3.11 (3.10 marche aussi si vous avez des soucis natifs sur macOS)
 - pip et virtualenv (intégrés à Python)
 - Accès Internet (téléchargement du modèle d’embedding la première fois)

# Installation : 

    # Cloner le repo
    git clone <URL_DU_REPO>.git
    cd RAG_Matisse

    # Créer/activer l'environnement virtuel
    python -m venv .venv
    source .venv/bin/activate         # PowerShell: .venv\Scripts\Activate.ps1

    # Installer les dépendances
    pip install -U pip
    pip install -r requirements.txt


# Configuration : 

    # Clé API (obligatoire)
    export MISTRAL_API_KEY="votre_cle_ici"

    # Modèle Mistral (optionnel)
    export MODEL="mistral-small-latest"

    # Base API (optionnel)
    export API_BASE="https://api.mistral.ai"

    # Stabilité (recommandé sur macOS / Intel)
    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

Astuce : pour éviter de retaper à chaque fois, mettez ces lignes dans un fichier env.local puis :
source env.local

# Préparer/Mettre à jour l’index
Si vous avez des documents à indexer (PDF/TXT/HTML dans data/), lancez :
python embed_index.py

Les embeddings sont générés une fois puis stockés (FAISS + métadonnées).
Le premier run télécharge le modèle d’embedding (peut prendre 1–2 min).

# Lancer l'interface Streamlit 

    streamlit run app.py --server.fileWatcherType=watchdog --server.port=8502

Ouvrez le lien donné par Streamlit (ex. http://localhost:8502).
Si vous êtes sur un autre OS/CPU, vous pouvez omettre --server.fileWatcherType=watchdog

# Utilisation en ligne de commande (optionnel)

    Tester la recherche/RAG en CLI :

    # Exemple 1 : petite requête
    python query.py "En quelle année le musée a ouvert ses portes ?"

    # Exemple 2 : script RAG direct (si un main est présent)
    python rag_answer.py

# Arborescence 

RAG_Matisse/
├─ app.py                # UI Streamlit
├─ config.py             # Config globale (noms de modèles, chemins, top_k, etc.)
├─ embed_index.py        # Construction / mise à jour de l’index FAISS
├─ query.py              # Requête RAG simple en CLI
├─ rag_answer.py         # Récupération contexte + appel Mistral
├─ text_chunker.py       # Découpage de textes (chunking)
├─ scrap.py              # (optionnel) scripts de collecte
├─ indexes/              # Index FAISS + métadonnées (générés)
├─ data/                 # Vos documents sources (PDF/TXT/HTML…)
├─ requirements.txt
└─ README.md

# Dépannage 

Segmentation fault / crash au démarrage
    Assurez-vous d’avoir exporté :

    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES



