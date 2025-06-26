import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
import os

# Chargement des données (optionnel si utilisé uniquement pour les dossiers)
@st.cache_data
def load_data():
    df = pd.read_parquet("data/echantillon_1000_hs_2024_TOC.parquet")
    df = df.rename(columns={"numdossier_new":"numdossier"})
    df= df.set_index("numdossier")
    return df

df = load_data()
liste_dossiers = sorted(df.index.unique().tolist())

# Chargement du vector store
@st.cache_resource
def load_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = Chroma(embedding_function=embedder,persist_directory="./chroma_db")
    return vector_store

vector_store = load_vectorstore()

# Interface utilisateur Streamlit
st.title("🔎 Exploration RAG : Visualisation du Retrieval")

num_dossier = st.selectbox("Sélectionnez un dossier :", liste_dossiers, index=0)

k = st.slider("Nombre de chunks retournés (k)", min_value=1, max_value=20, value=10)

question = st.text_input("Posez une question :")

if question and num_dossier:
    # Création du retriever
    retriever = vector_store.as_retriever(search_kwargs={
        "k": k,
        "filter": {"numdossier": num_dossier}
    })

    # Récupération des documents
    with st.spinner("Recherche en cours..."):
        results = retriever.get_relevant_documents(question)

    # Affichage des résultats
    st.success(f"{len(results)} chunk(s) retourné(s) pour le dossier `{num_dossier}`")

    for i, doc in enumerate(results, 1):
        with st.expander(f"Chunk {i}"):
            st.markdown(f"**Contenu :**\n\n{doc.page_content}")
            st.markdown("**Métadonnées :**")
            st.json(doc.metadata)
