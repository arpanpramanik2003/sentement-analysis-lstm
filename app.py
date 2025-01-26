import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved LSTM model and tokenizer
model = load_model('sentiment_lstm_model.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Streamlit app UI
st.title("IMDB Movie Review Sentiment Analysis")

# User input for movie review
review = st.text_area("Enter a movie review:")

# Predict sentiment on button click
if st.button("Analyze Sentiment"):
    if review:
        # Preprocess the input review
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        
        # Predict sentiment
        prediction = model.predict(padded_sequence)
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"

        # Display the sentiment result
        st.write("### Sentiment Analysis Result:")
        st.success(sentiment)
    else:
        st.error("Please enter a review.")