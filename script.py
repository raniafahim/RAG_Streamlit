import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os


# Chargement des données (optionnel si utilisé uniquement pour les dossiers)
@st.cache_data
def load_data():
    df = pd.read_parquet("/home/onyxia/work/Decoupage/data/echantillon_1000_hs_2024_TOC.parquet")
    df = df.rename(columns={"numdossier_new":"numdossier"})
    df= df.set_index("numdossier")
    return df

df = load_data()
liste_dossiers = sorted(df.index.unique().tolist())

# Chargement du vector store
@st.cache_resource
def load_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(embedding_function=embedder ,persist_directory="./chroma_db")
    return vectorstore


vectorstore = load_vectorstore()



st.set_page_config(page_title="📚 RAG Visualizer", layout="wide")
st.title("🔍 Visualisation des chunks sélectionnés")

# 👉 Extraction des numéros de dossier disponibles
# Si pas possible via vectorstore.get(), tu peux préextraire les num_dossier
# et les stocker dans un fichier .json ou .csv pour charger ici
try:
    all_numdossiers = liste_dossiers
except Exception:
    st.error("Impossible de lire les numéros de dossier. Fournir un fichier externe.")
    all_numdossiers = []

# Interface utilisateur
num_dossier = st.selectbox("Numéro de dossier :", all_numdossiers)
k = st.slider("Nombre de chunks à afficher (k)", min_value=1, max_value=20, value=5)
question = st.text_input("❓ Poser une question")

# Bouton d'action
if st.button("🔎 Voir les chunks sélectionnés") and question and num_dossier:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"numdossier": num_dossier}}
    )
    docs: list[Document] = retriever.invoke(question)

    st.subheader("📄 Chunks récupérés (contexte utilisé)")

    for i, doc in enumerate(docs):
        st.markdown(f"### 🔹 Chunk {i+1}")
        st.markdown(f"`ID:` {doc.metadata['id']}")
        st.markdown(f"`NumDossier:` {num_dossier}")
        
        st.markdown("---")
        st.write(doc.page_content)
        st.markdown("---")


