import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from gensim.models import Word2Vec

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
        placeholder="Example: Aman is the HOD of HIT and loves NLP")

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Word2Vec"
    ]
)

# HELPER FUNCTION FOR WORD2VEC
def preprocess_for_w2v(text):
    doc = nlp(text.lower())
    tokens = [
        token.text
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens

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

    
    # WORD2VEC
    elif option == "Word2Vec":
        st.subheader("Word2Vec Output")

        tokens = preprocess_for_w2v(text)

        if len(tokens) < 2:
            st.warning("Please enter more meaningful text for Word2Vec.")
        else:
            st.markdown("### Tokens Used for Training")
            st.write(tokens)

            # Train Word2Vec Model
            model = Word2Vec(
                sentences=[tokens],
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )

            selected_word = st.selectbox(
                "Select a word to view its vector",
                tokens
            )

            vector = model.wv[selected_word]

            st.markdown("### Word Vector (First 10 Dimensions)")
            st.write(vector[:10])

            st.markdown("### Vector Dimension")
            st.write(len(vector))

            st.markdown("### Similar Words")
            similar_words = model.wv.most_similar(selected_word)

            df_sim = pd.DataFrame(
                similar_words,
                columns=["Word", "Similarity Score"]
            )

            st.dataframe(df_sim, use_container_width=True)

            # Visualization (Similarity Scores)
            fig, ax = plt.subplots()
            ax.barh(
                df_sim["Word"],
                df_sim["Similarity Score"]
            )
            ax.set_title("Word Similarity Scores")
            ax.invert_yaxis()

            st.pyplot(fig)