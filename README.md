
# 🧠 RAG & Découpage de Documents

Ce projet implémente une approche **RAG (Retrieval-Augmented Generation)**, combinant **recherche de documents** et **génération de texte**. L’objectif est d’améliorer la pertinence des réponses générées grâce à une **méthodologie de découpage optimisée des documents sources**.

---

## ⚠️ Prérequis important : base vectorielle

Avant toute exécution, **vous devez manuellement télécharger la base `chroma_db_article`**, utilisée pour l’indexation vectorielle.

📦 **Cette base n’est pas incluse dans le dépôt** (trop volumineuse pour un `git push`) — elle doit être ajoutée manuellement. Pour cela, il faut lancer le notebook 'creacte_vectorestore_article' du dépôt RAG. 

🗂️ **Structure attendue du projet :**

```
project/
├── chroma_db_article/   ← 📂 À placer ici
├── Accueil.py
├── setup.sh
├── .venv/
├── README.md
└── ...
```

---

## 🧪 Mise en place de l’environnement

1. Lancer le script d’installation :

```bash
bash setup.sh
```

2. Activer l’environnement virtuel :

```bash
source .venv/bin/activate
```

---

## 🚀 Lancer l'application

Une fois l’environnement activé et la base vectorielle en place :

```bash
streamlit run Accueil.py
```

---

## 🧩 Fonctionnalités principales

* 🔍 **Retrieval vectoriel** via ChromaDB
* 📚 **Découpage intelligent** des documents pour un meilleur embedding
* 🤖 **RAG (Retrieval-Augmented Generation)** avec intégration d’un modèle de génération
* 🖥️ Interface simple via **Streamlit**

