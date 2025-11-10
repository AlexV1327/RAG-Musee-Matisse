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

q = st.text_input("Pose ta question", "Quels sont les horaires d'ouverture ?")
if st.button("Répondre"):
    with st.spinner("Recherche des passages pertinents…"):
        ctx, sources = gather_context(q)
    prompt = build_prompt(q, ctx)
    with st.spinner("Génération de la réponse…"):
        try:
            ans = answer_with_mistral(prompt)
        except Exception as e:
            st.error(str(e))
            st.stop()
    st.subheader("Réponse")
    st.write(ans)

    if sources:
        st.subheader("Sources")
        uniq = list(dict.fromkeys(sources))
        for s in uniq:
            if " — " in s:
                title, url = s.split(" — ", 1)
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {s}")

    with st.expander("Contexte envoyé au LLM (debug)"):
        st.code(ctx)
