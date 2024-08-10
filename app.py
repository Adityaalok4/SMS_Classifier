import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading the saved vectorizer and Naive Bayes model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb')) 

#transform_text function for the text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords') 

ps = PorterStemmer()

# Removed extra space here
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) # Tokenize the input text and assign it to 'text'

#removing special character and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]

#removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

#Applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Streamlit code
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    #preprocess the input message
    transformed_sms = transform_text(input_sms)

    #vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])

    #make Prediction 
    result = model.predict(vector_input)[0]
    # Fixed indentation here:
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")