import streamlit as st
import pickle
import string 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
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
        if i not in stopwords.words("English") and i not in string.punctuation:
               y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)         
tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))  
st.markdown("© 2023 MadhuKasa. All rights reserved.")
st.title("Email/SMS_Spam_Detection") 
input_sms = st.text_area("Enter the Message Here")
if st.button("Predict"):
     transformed_text = transform_text(input_sms)
     vector_input = tfid.transform([transformed_text])
     result = model.predict(vector_input)[0]
     if result==1:
             st.header("Spam")
     else:
             st.header("Not Spam")     
