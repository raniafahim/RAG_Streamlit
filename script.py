import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document

# 📌 Chargement de l'embedding et du vecteurstore
MODEL_NAME_EMBEDDER = "BAAI/bge-m3"
embedder = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDER)

vectorstore = Chroma(
    embedding_function=embedder,
    persist_directory="./chroma_db"
)

st.set_page_config(page_title="📚 RAG Visualizer", layout="wide")
st.title("🔍 Visualisation des chunks sélectionnés")

# 👉 Extraction des numéros de dossier disponibles
# Si pas possible via vectorstore.get(), tu peux préextraire les num_dossier
# et les stocker dans un fichier .json ou .csv pour charger ici
try:
    # Hack pour récupérer des dossiers (si accessible)
    all_docs = vectorstore.get()["documents"]
    all_numdossiers = sorted(set([d.metadata["num_dossier"] for d in all_docs]))
except Exception:
    st.error("Impossible de lire les numéros de dossier. Fournir un fichier externe.")
    all_numdossiers = []

# 🎛️ Interface utilisateur
num_dossier = st.selectbox("Numéro de dossier :", all_numdossiers)
k = st.slider("Nombre de chunks à afficher (k)", min_value=1, max_value=20, value=5)
question = st.text_input("❓ Poser une question")

# Bouton d'action
if st.button("🔎 Voir les chunks sélectionnés") and question and num_dossier:

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"num_dossier": num_dossier}}
    )
    docs: list[Document] = retriever.get_relevant_documents(question)

    st.subheader("📄 Chunks récupérés (contexte utilisé)")

    for i, doc in enumerate(docs):
        st.markdown(f"### 🔹 Chunk {i+1}")
        st.markdown(f"`ID:` {doc.metadata.get('id', 'N/A')}")
        st.markdown(f"`NumDossier:` {doc.metadata.get('num_dossier', 'N/A')}")
        st.markdown("---")
        st.write(doc.page_content)
        st.markdown("---")

    if st.checkbox("🧠 Générer une réponse à partir de ces chunks ?"):
        llm = HuggingFaceHub(repo_id="google/flan-t5-large")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(question)
        st.success("Réponse générée :")
        st.write(response)
