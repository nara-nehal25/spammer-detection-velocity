
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd 





st.write("# Spammer Detection and Fake User Identification  ")
st.sidebar.success("Select a function above.")


@st.cache_data
def naiveBayes():
    global classifier
    global cvv
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(),stop_words = "english", lowercase = True)
    dirname2=st.write('Naive Bayes Classifier loaded')



    
    





if st.button('Load Naive Bayes to analyse tweets'):
     naiveBayes()

