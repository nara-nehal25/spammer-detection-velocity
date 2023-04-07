from pathlib import Path
import streamlit as st
import json
import os
import re
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle

global filename

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

st.write("## Determine spam and fake users") 
st.sidebar.success("Select a function above.")

def fakeDetection(): 
    global root,cvv,classifier
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(),stop_words = "english", lowercase = True)
    dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'
    for root, dirs, files in os.walk(filename):
        for fdata in files:
            with open(root+"/"+fdata, "r") as file:
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
                st.write("Username : "+username+"\n")
                st.write("Tweet Text : "+textdata)
                st.write("Retweet Count : "+str(retweet)+"\n")
                st.write("Following : "+str(following)+"\n")
                st.write("Followers : "+str(followers)+"\n")
                st.write("Reputation : "+str(density)+"\n")
                st.write("Hashtag : "+str(hashtag)+"\n")
                st.write("Tweet Words Length : "+str(len(words))+"\n")
                test = cvv.fit_transform([textdata])
                spam = classifier.predict(test)
                cname = 0
                fake = 0
                if spam == 0:
                     st.write('"Tweet text contains : Non-Spam Words\n"')
                     cname = 0
                else:
                    st.write("Tweet text contains : Spam Words\n")
                    cname = 1
                if followers < following:
                     st.write("Twitter Account is Fake\n")
                     fake = 1
                else:
                     st.write("Twitter Account is Genuine\n")
                     fake = 0
                value = str(replies)+","+str(retweet)+","+str(following)+","+str(followers)+","+str(density)+","+str(hashtag)+","+str(fake)+","+str(cname)+"\n"
                dataset+=value

    f = open("features.txt", "w")
    f.write(dataset)
    f.close()



filename=st_directory_picker()
if st.button('Run Random forest for fake accounts'):
    fakeDetection()






    