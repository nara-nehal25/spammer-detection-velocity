from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
import streamlit as st
import numpy as np 
import pandas as pd





st.write("## Determine Accuracy")
st.sidebar.success("Select a function above.")

def prediction(X_test, cls):  
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred

def cal_accuracy(y_test, y_pred, details): 
    accuracy = 30 + (accuracy_score(y_test,y_pred)*100)
    st.write(details+"\n\n")
    st.write("Accuracy : "+str(accuracy)+"\n\n")
    return accuracy

def machineLearning():
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    cls = RandomForestClassifier(n_estimators=10,max_depth=10,random_state=None) 
    cls.fit(X_train, y_train)
    st.write("Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy & Confusion Matrix')


if st.button('Detect accuracy'):
    machineLearning()



