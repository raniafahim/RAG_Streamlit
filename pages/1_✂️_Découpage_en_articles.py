import streamlit as st
import re
import pandas as pd
import unidecode 



st.set_page_config("Découpage en articles", page_icon="✂️")
st.title("Découpage en articles")

st.markdown(
    """
    ## Pourquoi découper en articles ?

    La découpe en articles permet de structurer les documents d'accords en **paragraphes cohérents**, correspondant aux différentes sections thématiques.  
    Cette granularité facilite l'analyse, la lecture, ainsi que le travail statistique ou automatique sur des éléments bien délimités.

    ## Stratégie de découpe

    - Nous réutilisons les **sommaires extraits d'une étude précédente**, qui ont permis de construire un ensemble de titres représentatifs.  
    - Ces sommaires servent ici à **segmenter le texte original** en parties correspondant aux articles ou sections du document.  
    - Pour garantir la cohérence, lorsqu'un titre est suivi immédiatement d'un sous-titre, ou lorsque deux segments extraits sont trop courts (moins de 100 tokens), nous les **concaténons** afin d'éviter des fragments trop petits et peu significatifs.
    """
)



def normalize(text):
    return unidecode.unidecode(text.lower().strip())


def clean_summary_titles(summary_titles):
    cleaned = []
    for title in summary_titles:
        stripped = title.strip()
        if not stripped:
            continue  # Titre vide
        if re.fullmatch(r"[_\-–—=~\.]{3,}", stripped):
            continue  
        cleaned.append(title)
    return cleaned

def split_text_with_titles(text, summary):
    """
    Découpe le texte en segments basés sur les titres du sommaire, après normalisation,
    en respectant strictement leur ordre d'apparition attendu.
    Retourne une liste de (titre_fusionné, texte_segmenté).
    """
    split_texts = []
    merged_titles = []

    positions = []

    normalized_text = normalize(text)
    lower_text = text.lower()

    position_courante = 0

    for sentence in summary:
        norm_sentence = normalize(sentence)
        pos_norm = normalized_text.find(norm_sentence, position_courante)
        if pos_norm != -1:
            pos_real = lower_text.find(sentence.lower(), position_courante)
            if pos_real != -1:
                positions.append(pos_real)
                position_courante = pos_real + 1

    if not positions:
        # Pas de découpage, titre "Document complet"
        return [("Document complet", text)]

    positions = sorted(set(positions))
    positions.insert(0, 0)
    positions.append(len(text))

    i = 0
    while i < len(positions) - 1:
        start = positions[i]
        end = positions[i + 1]

        # Calculer les titres concernés par ce segment
        # Quand i=0, aucun titre, on peut mettre "Intro" ou vide
        if i == 0:
            titles_group = []
        else:
            # i-1 correspond à l'index du titre dans summary (car positions contient un 0 en début)
            titles_group = [summary[i-1]]

        # Vérification de fusion avec le segment suivant
        if (end - start) < 150 and i + 2 < len(positions):
            # Fusionner textes
            next_end = positions[i + 2]

            # Fusionner titres associés aux 2 segments
            if i == 0:
                titles_group = []
            else:
                # titres des deux segments fusionnés
                titles_group = [summary[i-1], summary[i]]

            merged_title = " - ".join(titles_group)
            split_texts.append(text[start:next_end].strip())
            merged_titles.append(merged_title)
            i += 2
        else:
            # Pas de fusion, un seul titre ou intro
            if i == 0:
                merged_titles.append("Introduction")
            else:
                merged_titles.append(summary[i-1])

            split_texts.append(text[start:end].strip())
            i += 1

    return list(zip(merged_titles, split_texts))


@st.cache_data
def load_data():
    df = pd.read_parquet("/home/onyxia/work/Decoupage/data/echantillon_1000_hs_2024_TOC.parquet")
    df = df.rename(columns={"numdossier_new": "numdossier"})
    df = df.set_index("numdossier")
    return df

df = load_data()
liste_dossiers = sorted(df.index.unique().tolist())


df["extracted_summary"] = df.apply(
    lambda row: clean_summary_titles(row["extracted_summary"]),
    axis=1
)

df["section_dict"] = df.apply(
    lambda row: split_text_with_titles(row["accorddocx"], row["extracted_summary"]),
    axis=1
)

num_dossier = st.selectbox("**Numéro de dossier :**", liste_dossiers)
if st.button("**🔄 Découper l'accord par article**") and num_dossier: 
    st.markdown("### ✂️ Découpage par article")
    st.subheader(f"📄 Accord n° {num_dossier}")
    # Récupération des données
    texte = df.loc[num_dossier, "accorddocx"]
    summary = df.loc[num_dossier, "extracted_summary"]
    sections = df.loc[num_dossier, "section_dict"]
    for title, content in sections:
        anchor = normalize(title).replace(' ', '-')
        with st.expander(f"📝{title}"):
            st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
            st.write(content)
                
