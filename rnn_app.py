import numpy as np
import tensorflow as tf  
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index= imdb.get_word_index()
reverse_word_index= {value : key for key ,  value in word_index.items()}

model= load_model("imdb_rnn_model.keras" )

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words= text.lower().split()
    encoded_review= [word_index.get(w, 2) for w in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def prediction_sentiment(review):
    processed_input= preprocess_text(review)
    prediction= model.predict(processed_input)

    sentiment= "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment , prediction

## streamlit app
import streamlit as st
st.title("IMDB Movie Review analysis")
st.write("Enter a movie review")

user_input= st.text_area("movie review")

if st.button("classify"):
    sentiment, prediction= prediction_sentiment(user_input)
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"prediction: {prediction[0][0]:.2f}")

