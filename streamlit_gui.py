from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st

st.title("Sentiment analysis project")
st.info("Please enter the sentiment that you need to check about")

# Define text globally
global text
text = st.text_input("Enter the sentiment")
print(text)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

common_words = set(['game', 'com', 'unk', 'like'])
stop_words = set(stopwords.words('english')).union(common_words)
lem = WordNetLemmatizer()

def remove_stop_words(txt):
    return ' '.join([x for x in txt.split() if x not in stop_words])

def remove_punc(txt):
    text_non_punct = "".join([char for char in txt if char not in string.punctuation])
    return text_non_punct

def remove_digit(txt):
    text_non_digit = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", '', txt).strip()
    return text_non_digit

def lemmatizing(txt):
    lemmatized = [lem.lemmatize(word, pos='v') for word in txt]
    return lemmatized

def predict():
    # Declare text as global to access the global variable
    global text
    
    # Load the models
    tfidf_model = joblib.load('TFIDF_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    print(svm_model.n_features_in_)

    # Text processing steps
    text = text.lower() 
    print(text)
    text = remove_stop_words(text)
    print(text)
    text = remove_punc(text)
    print(text)
    text = remove_digit(text)
    print(text)
    text = word_tokenize(text)
    print(text)
    text = lemmatizing(text)
    print(text)
    text = ' '.join(text)
    print(text)

    # Transform the text using the TF-IDF model
    tfidf_test = tfidf_model.transform([text])

    # Predict using the loaded SVM model
    prediction = svm_model.predict(tfidf_test)
    print("Prediction:", prediction)
    st.subheader(f"Prediction Result: {prediction}")
# Call the prediction function
predict()

