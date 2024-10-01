import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Download stopwords and tokenizer (Only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stopwords.words("english")]
    return " ".join(tokens)

# Load and preprocess dataset
@st.cache(allow_output_mutation=True)
def load_data():
    # Use your actual path to load the dataset
    bot_dataset = pd.read_csv("C:\\Users\\91762\\Downloads\\Simple_Chat_bot-main\\Simple_Chat_bot-main\\topical_chat.csv")
    bot_dataset["processed_message"] = bot_dataset["message"].apply(preprocess_text)
    return bot_dataset

bot_dataset = load_data()

# Tokenization and Padding
@st.cache(allow_output_mutation=True)
def prepare_data(bot_dataset):
    X = bot_dataset["processed_message"]
    y = bot_dataset["sentiment"]

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=100, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_padded, y_encoded, tokenizer, label_encoder

X_padded, y_encoded, tokenizer, label_encoder = prepare_data(bot_dataset)

# Build the model
@st.cache(allow_output_mutation=True)
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=100),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='linear')
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    return model

model = build_model()

# Train the model if not trained already
if 'trained' not in st.session_state:
    model.fit(X_padded, y_encoded, epochs=3, batch_size=45, validation_split=0.1)
    st.session_state['trained'] = True

# Sentiment Prediction Function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
    sentiment_probabilities = model.predict(padded_sequence)
    predicted_sentiment_id = np.argmax(sentiment_probabilities)
    predicted_sentiment = label_encoder.inverse_transform([predicted_sentiment_id])[0]
    return predicted_sentiment

# Rule-based response generation
def generate_rule_based_response(predicted_sentiment):
    if predicted_sentiment == "Happy":
        response = "I'm glad to hear that you're feeling happy!"
    elif predicted_sentiment == "Sad":
        response = "I'm sorry to hear that you're feeling sad. Is there anything I can do to help?"
    else:
        response = "I'm here to chat with you. How can I assist you today?"
    return response

# Pattern-based response
def generate_pattern_response(user_input):
    patterns = {
        "hello": "Hello! How can I assist you today?",
        "how are you": "I'm just a chatbot, but I'm here to help! How can I assist you?",
        "help": "Sure, I'd be happy to help. What do you need assistance with?",
        "bye": "Goodbye! If you have more questions in the future, feel free to ask.",
    }

    for pattern, response in patterns.items():
        if pattern in user_input.lower():
            return response
    return generate_rule_based_response(predict_sentiment(user_input))

# Streamlit App Interface
def main():
    st.title("Chatbot with Sentiment Analysis")
    
    # Chatbot introduction
    st.write("Welcome to the chatbot! Ask me anything or tell me how you feel.")
    
    # Input from user
    user_input = st.text_input("You: ", "")
    
    if user_input:
        # Generate bot response
        bot_response = generate_pattern_response(user_input)
        st.write(f"Bot: {bot_response}")

if __name__ == "__main__":
    main()
