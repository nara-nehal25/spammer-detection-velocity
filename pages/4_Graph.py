from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import json
import os
import re
import pickle as cpickle


global filename


st.write("# Graph Detection  ")
st.sidebar.success("Select a function above.")

def st_directory_picker(initial_path=Path()):

    st.markdown("#### Directory picker")

    if "path" not in st.session_state:
        st.session_state.path = initial_path.absolute()

    manual_input = st.text_input("Selected directory:", st.session_state.path)

    manual_input = Path(manual_input)
    if manual_input != st.session_state.path:
        st.session_state.path = manual_input
        st.experimental_rerun()

    _, col1, col2, col3, _ = st.columns([3, 1, 3, 1, 3])

    with col1:
        st.markdown("#")
        if st.button("⬅️") and "path" in st.session_state:
            st.session_state.path = st.session_state.path.parent
            st.experimental_rerun()

    with col2:
        subdirectroies = [
            f.stem
            for f in st.session_state.path.iterdir()
            if f.is_dir()
            and (not f.stem.startswith(".") and not f.stem.startswith("__"))
        ]
        if subdirectroies:
            st.session_state.new_dir = st.selectbox(
                "Subdirectories", sorted(subdirectroies)
            )
        else:
            st.markdown("#")
            st.markdown(
                "<font color='#FF0000'>No subdir</font>", unsafe_allow_html=True
            )

    with col3:
        if subdirectroies:
            st.markdown("#")
            if st.button("➡️") and "path" in st.session_state:
                st.session_state.path = Path(
                    st.session_state.path, st.session_state.new_dir
                )
                st.experimental_rerun()

    return st.session_state.path

def graph():
     global total,fake_acc,spam_acc,root,cvv,classifier
     classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
     cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
     cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(),stop_words = "english", lowercase = True)
     total = 0
     fake_acc = 0
     spam_acc = 0
     dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'
     for root, dirs, files in os.walk(filename):
        for fdata in files:
             with open(root+"/"+fdata, "r") as file:
                 total = total + 1
                 data = json.load(file)
                 textdata = data['text'].strip('\n')
                 textdata = textdata.replace("\n"," ")
                 textdata = re.sub('\W+',' ', textdata)
                 retweet = data['retweet_count']
                 followers = data['user']['followers_count']
                 density = data['user']['listed_count']
                 following = data['user']['friends_count']
                 replies = data['user']['favourites_count']
                 hashtag = data['user']['statuses_count']
                 username = data['user']['screen_name']
                 words = textdata.split(" ")
                 test = cvv.fit_transform([textdata])
                 spam = classifier.predict(test)
                 cname = 0
                 fake = 0
                 if spam == 0:
                     cname = 0
                 else:
                      spam_acc = spam_acc + 1
                      
                 if followers < following:
                     
                     fake_acc = fake_acc + 1
                
                 
                     
    
     height = [total,spam_acc]
     bars = ('Total Twitter Accounts', 'Spam Content Tweets')
     y_pos = np.arange(len(bars))
     fig, ax = plt.subplots()
     plt.xticks(y_pos,bars)
     ax.bar(y_pos,height)
     st.pyplot(fig)

filename=st_directory_picker()
if st.button('Detection Graph'):
    graph()