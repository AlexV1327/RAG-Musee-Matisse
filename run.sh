# Exporter la clé API 

source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE   # seulement tant que le conflit OpenMP n’est pas corrigé
export OMP_NUM_THREADS=1

python -m streamlit run app.py
