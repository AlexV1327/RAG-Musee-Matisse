# app.py — Streamlit UI pour ton RAG
import os
import streamlit as st

from rag_answer import gather_context, build_prompt, answer_with_mistral
from config import EMB_MODEL_NAME, TOP_K, MAX_CONTEXT_CHARS, API_MODEL

st.set_page_config(page_title="RAG Matisse", page_icon="🎨")
st.title("🎨 RAG — Musée Matisse (POC)")

with st.sidebar:
    st.header("Paramètres")
    st.write(f"**Embeddings** : `{EMB_MODEL_NAME}`")
    st.write(f"**TOP_K** : {TOP_K}")
    st.write(f"**Max contexte** : {MAX_CONTEXT_CHARS}")
    st.write(f"**LLM** : `{API_MODEL}`")
    has_key = bool(os.getenv("API_KEY") or os.getenv("MISTRAL_API_KEY"))
    st.write(f"**API Key** : {'✅' if has_key else '❌'}")

st.caption("Posez une question sur le musée (horaires, tarifs, accès, expositions, histoire…).")

q = st.text_input("Votre question", value="Quels sont les horaires d’ouverture du musée ?")
if st.button("Répondre") and q.strip():
    with st.spinner("Récupération des passages…"):
        ctx, sources = gather_context(q.strip())

    if not ctx or ctx.strip() == "[Contexte vide]":
        st.warning("Aucun passage pertinent trouvé dans l'index.")
    else:
        prompt = build_prompt(q.strip(), ctx)
        with st.spinner("Génération de la réponse…"):
            try:
                answer = answer_with_mistral(prompt)
            except Exception as e:
                st.error(f"Erreur Mistral: {e}")
                answer = None

        if answer:
            st.subheader("Réponse")
            st.markdown(answer)

            if sources:
                st.subheader("Sources")
                # dédup en gardant l'ordre
                uniq = list(dict.fromkeys(sources))
                for s in uniq:
                    # s est au format "Titre — URL"
                    if " — " in s:
                        title, url = s.split(" — ", 1)
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {s}")

        with st.expander("Contexte envoyé au LLM (debug)"):
            st.code(ctx)
