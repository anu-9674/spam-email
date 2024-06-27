import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pathlib import Path
current_directory= Path(__file__).resolve().parent
tfidf_path=current_directory/"vectorizer.pkl"
model_path=current_directory/"model.pkl"
tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

ps=PorterStemmer()

# tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
# model = pickle.load(open("model.pkl", 'rb'))


st.title("SPAM EMAIL CLASSIFIER")


#1.Preprocess- we need the transform_text function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
       y.append(ps.stem(i))

    return " ".join(y)

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    #2.Vectorize
    vector_input = tfidf.transform([transform_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #Display
    if result == 0:
        st.header("Not Spam")
    else:
        st.header("Spam")


