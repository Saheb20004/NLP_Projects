import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# DOWNLOAD NLTK DATA
nltk.download("punkt")
nltk.download("stopwords")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing",
    layout="wide"
)

# APP TITLE
st.title("NLP Preprocessing App")
st.write("Tokenization, Text Cleaning, Stemming, Lemmatization, and Bag of Words")


# USER INPUT
text = st.text_area("Enter text for NLP processing", height=150,
        placeholder="Example: Krishnendu is a student of HIT and he loves his family very much.")

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "TF-IDF"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    
    # TOKENIZATION
    elif option == "Tokenization":
        st.subheader("Tokenization Output")
        col1, col2, col3 = st.columns(3)

        # Sentence Tokenization
        with col1:
            st.markdown("### Sentence Tokenization")
            sentences = sent_tokenize(text)
            st.write(sentences)

        # Word Tokenization
        with col2:
            st.markdown("### Word Tokenization")
            words = word_tokenize(text)
            st.write(words)

        # Character Tokenization
        with col3:
            st.markdown("### Character Tokenization")
            characters = list(text)
            st.write(characters)

    
    # TEXT CLEANING
    elif option == "Text Cleaning":
        st.subheader("Text Cleaning Output")

        # Convert text to lowercase
        text_lower = text.lower()

        # Remove punctuation & numbers
        cleaned_text = "".join(ch for ch in text_lower if ch not in string.punctuation and not ch.isdigit())

        # Remove stopwords using spaCy
        doc = nlp(cleaned_text)
        final_words = [token.text for token in doc if not token.is_stop and token.text.strip() != ""]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    
    # STEMMING
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)

        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        # Apply stemming
        porter_stem = [porter.stem(word) for word in words]
        lancaster_stem = [lancaster.stem(word) for word in words]

        # Comparison table
        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": porter_stem,
            "Lancaster Stemmer": lancaster_stem
        })

        st.dataframe(df, use_container_width=True)

    # LEMMATIZATION
    elif option == "Lemmatization":
        st.subheader("Lemmatization using spaCy")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

      # TF-IDF
    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation")

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text])

        vocab = vectorizer.get_feature_names_out()
        freq = X.toarray()[0]

        df = pd.DataFrame({
            "Word": vocab,
            "TF-IDF Score": freq
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.markdown("### TF-IDF Score Table")
        st.dataframe(df, use_container_width=True)

        
        # BAR CHART (TOP-N WORDS)
        st.markdown("### TF-IDF Score Distribution (Top 10)")

        top_n = 10
        df_top = df.head(top_n)

        fig, ax = plt.subplots()
        ax.bar(
            df_top["Word"],
            df_top["TF-IDF Score"]
        )
        plt.xticks(rotation=45)

        st.pyplot(fig)
     