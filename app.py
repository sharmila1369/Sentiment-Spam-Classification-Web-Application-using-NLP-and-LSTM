import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request
import pandas as pd
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Masking, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download nltk data
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# ===== PARAMETERS (MATCH TRAINING) =====
VOCAB_SIZE = 2000
MAX_LEN = 150
EMBEDDING_DIM = 64

# ===== LOAD DATASET TO BUILD TOKENIZER =====
df = pd.read_csv("IMDB Dataset.csv", encoding="unicode_escape")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([i for i in text if i not in string.punctuation])
    text = ' '.join([
        lemmatizer.lemmatize(i) for i in text.split()
        if i not in stop_words
    ])
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# ===== BUILD TOKENIZER (NO tokenizer.pkl NEEDED) =====
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df['cleaned_review'])

# ===== RECREATE YOUR TRUE MODEL ARCHITECTURE =====
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
model.add(Masking(mask_value=0.0))
model.add(Bidirectional(LSTM(64, return_sequences=True)))   # ← FIXED
model.add(Bidirectional(LSTM(32, return_sequences=False))) # ← FIXED
model.add(Dense(2, activation='sigmoid'))

# ===== LOAD WEIGHTS (WILL MATCH NOW) =====
model.load_weights('sentiment_lstm_model.h5')

labels = ['Negative 😞', 'Positive 😊']

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    prediction = None
    user_review = ""

    if request.method == 'POST':
        user_review = request.form['review']

        # Clean input
        text_clean = clean_text(user_review)

        # Convert text to numbers
        seq = tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        pred = model.predict(padded)
        prediction = labels[np.argmax(pred)]

    return render_template(
        'index.html',
        prediction=prediction,
        user_review=user_review
    )

if __name__ == '__main__':
    app.run(debug=False)
