# Sentiment-Spam-Classification-Web-Application-using-NLP-and-LSTM
# 📩 SENTIMENT / SPAM CLASSIFICATION PROJECT
## 1️⃣ Project Title

Sentiment & Spam Classification Web Application using NLP and LSTM

## 2️⃣ Abstract / Overview

This project is a Flask-based web application that classifies a given text message as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and a Deep Learning (LSTM) model.

The system performs:

Text cleaning and preprocessing

Tokenization of words

Prediction using a pre-trained LSTM model

Displaying results through a user-friendly web interface

## 3️⃣ Objectives

The main objectives of this project are:

To analyze and preprocess textual data

To build a machine learning-based spam detection system

To integrate the trained model with a Flask web application

To allow users to input messages and get real-time predictions

## 4️⃣ Features of the Project

✔ Interactive web interface
✔ Text preprocessing (cleaning, tokenization, lemmatization)
✔ Pretrained LSTM deep learning model
✔ Fast and accurate predictions
✔ Flask backend integration

## 5️⃣ Technologies Used
Technology	Purpose
Python	Programming Language
Flask	Web Framework
TensorFlow / Keras	Deep Learning Model
NLTK	Natural Language Processing
Pandas	Data Handling
NumPy	Numerical Computation
HTML	Frontend Structure
CSS	Styling
## 6️⃣ Dataset Description

The dataset used in this project is:

SMS Spam Collection Dataset

It contains two columns:

label → spam / ham

message → actual SMS text

## 7️⃣ System Architecture

User Input → Message entered in web app

Preprocessing → Text cleaned and tokenized

Model Prediction → LSTM model predicts spam probability

Output Display → Result shown as Spam or Not Spam

## 8️⃣ Project Folder Structure
Spam-Classifier-Project/
│
├── app.py                # Main Flask application
├── create_tokenizer.py   # Script to create tokenizer
├── spam_lstm_model.h5    # Trained LSTM model
├── new_tokenizer.pkl     # Saved tokenizer file
├── spam.csv              # Dataset
│
├── templates/
│   └── index.html        # Frontend UI
│
├── static/
│   └── style.css         # Styling file
│
└── .venv/                # Virtual environment

## 9️⃣ How the System Works
Step 1: Data Preprocessing

Convert text to lowercase

Remove special characters

Remove stopwords

Apply lemmatization

Step 2: Tokenization

Convert words into numerical format using Tokenizer

Step 3: Model Prediction

Input is padded to fixed length

Passed to LSTM model

If probability > 0.5 → Spam

Else → Not Spam

# 🔟 How to Run the Project
Step 1: Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

Step 2: Install Required Packages
pip install flask pandas numpy nltk tensorflow scikit-learn

Step 3: Download NLTK Data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

Step 4: Create Tokenizer
python create_tokenizer.py

Step 5: Run Flask Application
python app.py

Step 6: Open in Browser
http://127.0.0.1:5000

## 1️⃣1️⃣ Sample Predictions
Input Message	Output
"Win a free iPhone now!"	Spam 🚫
"Can we meet tomorrow?"	Not Spam ✅
## 1️⃣2️⃣ Advantages

Helps filter unwanted spam messages

Reduces manual effort

Can be integrated with email systems

Scalable and extendable

## 1️⃣3️⃣ Limitations

Works best for English text

Accuracy depends on training data

May misclassify sarcastic messages

## 1️⃣4️⃣ Future Enhancements

Add confidence percentage in prediction

Deploy on cloud (AWS / Render / Railway)

Support multiple languages

Improve UI with charts and analytics

## 1️⃣5️⃣ Author

Venna Sharmilambika

B.Tech CSE (AI & Data Science)
Email: [sharmilambikavenna@gmail.com](mailto:sharmilambikavenna@gmail.com)

