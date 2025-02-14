import streamlit as st
import spacy
import spacy_streamlit
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from helper_modules.agent import summarize_text
import seaborn as sns
# Load spaCy models
nlp_small = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Load dataset
df = st.session_state.get("dataset")

if df is not None:
    # Identify text columns
    text_columns = [col for col in df.columns if df[col].dtype == "object"]

    # Find columns with average word count >= 5
    valid_columns = [
        col for col in text_columns 
        if df[col].dropna().apply(lambda x: len(str(x).split())).mean() >= 5
    ]

    if valid_columns:
        st.title("NLP Analysis of Reviews")
        st.write("Using spaCy & AI-powered summarization for text analysis.")

        # Select a column from valid ones
        selected_column = st.selectbox("Choose a column for analysis:", valid_columns)

        # Select a review from the chosen column
        selected_review = st.selectbox("Choose a review:", df[selected_column].dropna())

        df["sentiment_score"] = df[selected_column].dropna().apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
        # Process text with spaCy
        doc = nlp_small(selected_review)

        # Categorize sentiment
        df["sentiment_category"] = df["sentiment_score"].apply(
            lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
        )

        # Sentiment distribution plot
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df["sentiment_category"], palette={"Positive": "green", "Neutral": "gray", "Negative": "red"})
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Analysis Distribution")
        st.pyplot(fig)

        

        # AI-powered summarization
        if st.button("Generate Summary"):
            summary = summarize_text(selected_review)
            st.subheader("Summary")
            st.write(summary)

        # Visualization of NLP features
        spacy_streamlit.visualize_ner(doc, labels=nlp_small.get_pipe("ner").labels)

        # WordCloud by POS selection
        pos_options = ["NOUN", "ADJ", "VERB", "ALL"]
        selected_pos = st.selectbox("Select Part of Speech for WordCloud:", pos_options)

        if st.button("Generate WordCloud"):
            words = [token.text.lower() for token in doc if token.is_alpha and 
                     (selected_pos == "ALL" or token.pos_ == selected_pos)]
            wordcloud_text = " ".join(words)

            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(wordcloud_text)

            st.subheader("WordCloud")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    else:
        st.error("No text column found with an average word length of at least 5 words.")
else:
    st.error("Dataset not found! Please upload or set a dataset.")
