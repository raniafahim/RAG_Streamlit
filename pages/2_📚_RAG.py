import streamlit as st
import pandas as pd
import re
import unidecode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

st.set_page_config(page_title="RAG Visualizer", layout="wide", page_icon="📚")
st.title("📚 RAG Visualizer")

st.markdown(
    """
    Cette page permet de visualiser un processus de Retrieval-Augmented Generation (RAG) sur un corpus de 1000 accords portant sur les "Heures supplémentaires"
.
    **Étapes du processus :**  
    1. 📄 Affichage du découpage de l’accord  
    2. 🔎 Affichage des chunks sélectionnés par la recherche  
    3. ✅ Contexte final envoyé au LLM  
    """
)

# ------------------------- Caching des données -------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet("/home/onyxia/work/RAG_Streamlit/data/echantillon_1000_hs_2024_TOC.parquet")
    df = df.rename(columns={"numdossier_new": "numdossier"})
    df = df.set_index("numdossier")
    return df

@st.cache_resource
def load_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(embedding_function=embedder, persist_directory="./chroma_db_article")
    return vectorstore


# ------------------------- État de session -------------------------

if "step" not in st.session_state:
    st.session_state.step = 1
if "docs" not in st.session_state:
    st.session_state.docs = []
if "selected_chunks" not in st.session_state:
    st.session_state.selected_chunks = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_dossier" not in st.session_state:
    st.session_state.last_dossier = None

highlight_style = "background-color: #FFFE94; border-left: 6px solid #FFFE94; padding: 10px; margin-bottom: 20px; border-radius: 10px;"
def display_chunk(doc: Document, highlight=False, preview_words=30):
    full_content = doc.page_content
    titre = doc.metadata.get("title", "Sans titre").strip()  # <- nettoyage ici
    header = f"**✔️Chunk sélectionné : {titre}**" if highlight else f"{titre}"
    with st.expander(header):
        if highlight:
            st.markdown(f"<div style='{highlight_style}'>{full_content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(full_content)


def extract_index(metadata_id):
    return int(metadata_id.split("_")[1])
# ------------------------- Interface utilisateur -------------------------

df = load_data()
vectorstore = load_vectorstore()
liste_dossiers = sorted(df.index.unique().tolist())

num_dossier = st.selectbox("**Numéro de dossier :**", liste_dossiers)
k = st.slider("**Nombre de chunks à afficher (k)**", min_value=1, max_value=20, value=5)
question = st.text_input("**❓ Poser une question**")


# ------------------------- Retrieval automatique -------------------------

if st.button("**🔄 Lancer le Retrieval**") and question and num_dossier:
    st.session_state.step = 1
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"numdossier": num_dossier}}
    )
    selected_chunks = retriever.invoke(question)

    raw = vectorstore._collection.get(
        where={"numdossier": num_dossier},
        include=["documents", "metadatas"]
    )

    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in sorted(
            zip(raw["documents"], raw["metadatas"]),
            key=lambda x: extract_index(x[1]["id"])
        )
    ]

    st.session_state.docs = all_docs
    st.session_state.selected_chunks = selected_chunks
    st.session_state.last_question = question
    st.session_state.last_dossier = num_dossier
    st.success(f"{len(selected_chunks)} chunks récupérés pour le dossier {num_dossier}")

# ------------------------- Navigation étapes -------------------------

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("◀ Précédent", use_container_width=True) and st.session_state.step > 1:
        st.session_state.step -= 1
with col3:
    if st.button("▶ Suivant", use_container_width=True) and st.session_state.step < 3:
        st.session_state.step += 1

# ------------------------- Affichage des étapes -------------------------

if st.session_state.docs:
    if st.session_state.step == 1:
        st.subheader(f"📄 Accord n°{num_dossier} découpé ")
        for doc in st.session_state.docs:
            display_chunk(doc)

    elif st.session_state.step == 2:
        st.subheader("🔍 Chunks sélectionnés par le retrieval")
        selected_ids = {doc.metadata["id"] for doc in st.session_state.selected_chunks}
        for doc in st.session_state.docs:
            display_chunk(doc, highlight=(doc.metadata["id"] in selected_ids))

    elif st.session_state.step == 3:
        st.subheader("✅ Contexte envoyé au LLM")
        for doc in st.session_state.selected_chunks:
            display_chunk(doc, highlight=True)
