import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

st.set_page_config(page_title="📚 RAG Visualizer", layout="wide")
st.title("📚 RAG Visualizer")


# Initialisation de l'état
if "step" not in st.session_state:
    st.session_state.step = 1
if "docs" not in st.session_state:
    st.session_state.docs = []
if "selected_chunks" not in st.session_state:
    st.session_state.selected_chunks = []

# Style
highlight_style = "background-color: #FFFE94; border-left: 6px solid #FFFE94; padding: 10px; margin-bottom: 20px; border-radius: 10px;"

def display_chunk(doc: Document, highlight=False, preview_words=30):
    full_content = doc.page_content
    preview = " ".join(full_content.split()[:preview_words]) + "..."

    chunk_id = doc.metadata.get("id", "N/A")
    header = f"**✔️ Chunk sélectionné (ID: {chunk_id})**" if highlight else f"Chunk (ID: {chunk_id})"

    with st.expander(header):
        if highlight:
            st.markdown(
                f"<div style='{highlight_style}'>{full_content}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(preview)
            st.markdown(full_content)

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("◀ Précédent", use_container_width=True) and st.session_state.step > 1:
        st.session_state.step -= 1
with col3:
    if st.button("▶ Suivant", use_container_width=True) and st.session_state.step < 3:
        st.session_state.step += 1

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_parquet("/home/onyxia/work/Decoupage/data/echantillon_1000_hs_2024_TOC.parquet")
    df = df.rename(columns={"numdossier_new": "numdossier"})
    df = df.set_index("numdossier")
    return df

df = load_data()
liste_dossiers = sorted(df.index.unique().tolist())

# Chargement du vectorstore
@st.cache_resource
def load_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(embedding_function=embedder, persist_directory="./chroma_db")
    return vectorstore

vectorstore = load_vectorstore()

# Interface utilisateur
num_dossier = st.selectbox("**Numéro de dossier :**", liste_dossiers)
k = st.slider("**Nombre de chunks à afficher (k)**", min_value=1, max_value=20, value=5)
question = st.text_input("**❓ Poser une question**")

# Bouton d'action
if st.button("**🔎 Voir les chunks sélectionnés**") and question and num_dossier:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"numdossier": num_dossier}}
    )
    docs: list[Document] = retriever.invoke(question)
    raw = vectorstore._collection.get(
        where={"numdossier": num_dossier},
        include=["documents", "metadatas"]
    )
    
    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in sorted(
            zip(raw["documents"], raw["metadatas"]),
            key=lambda x: x[1].get("chunk_id", x[1].get("id", 0))  # tri par chunk_id si dispo, sinon par id
        )
    ]
    # Mémoriser les résultats
    st.session_state.docs = all_docs
    st.session_state.selected_chunks = docs  # ici on suppose que les "chunks sélectionnés" = ceux récupérés

    st.success(f"{len(docs)} chunks récupérés pour le dossier {num_dossier}")

# Affichage dynamique (si des chunks ont été chargés)
if st.session_state.docs:
    placeholder = st.empty()
    with placeholder.container():
        if st.session_state.step == 1:
            st.subheader(f"**📄 Découpage de l'accord {num_dossier}**")
            for doc in st.session_state.docs:
                display_chunk(doc)

        elif st.session_state.step == 2:
            st.subheader("**🔎 Retrieval : sélection des paragraphes**")
            selected_ids = {doc.metadata["id"] for doc in st.session_state.selected_chunks}
            for doc in st.session_state.docs:
                display_chunk(doc, highlight=(doc.metadata["id"] in selected_ids))

        elif st.session_state.step == 3:
            st.subheader("**✅ Contexte fourni au LLM**")
            for doc in st.session_state.selected_chunks:
                display_chunk(doc, highlight=True)



