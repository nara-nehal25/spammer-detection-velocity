import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from collections import OrderedDict
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings


warnings.filterwarnings('ignore')


st.title('fake account detector')
st.sidebar.success("Select a function above.")

filename=None
ErrorrateMeans = list()
AccuracyMeans = list()

filename=st.file_uploader("Choose a file for training")

def Knn():
    if filename:
        global AccuracyMeans,ErrorrateMeans
        df=pd.read_csv(filename) 
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        test = df[~msk]
        testing_data=test.values[:, 0:7]
        testing_data_labels=test.values[:, 8]
        features = train.values[:, 0:7]
        labels   = train.values[:, 8].astype('int')

        model3 = KNeighborsClassifier(n_neighbors=3)
        model3.fit(features,labels)
        predictions_model3 = model3.predict(testing_data)
    
        accuracy=accuracy_score(testing_data_labels, predictions_model3)*100
        AccuracyMeans.append(accuracy)
        error_rate=100-accuracy
        ErrorrateMeans.append(error_rate)
        precision=precision_score(testing_data_labels, predictions_model3)*100
        recall=recall_score(testing_data_labels, predictions_model3)*100
        
        st.write('3.K-Nearest Neighbors  :\n')
        st.write('Confusion Matrix :')
        st.write(confusion_matrix(testing_data_labels, predictions_model3)) 
        st.write('Accuracy Is : '+str(accuracy )+' %')
        st.write('Error Rate Is : '+str(error_rate)+' %')
        st.write('Precision Is : '+str(precision)+' %')
        st.write('Recall Is : '+str(recall)+' %\n\n')

        labels = ['Error Rate', 'Accuracy ']
        sizes = [error_rate,accuracy]

        explode = (0, 0.1)  
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90) 

        plt.title('KNN Algorithm')
        ax1.axis('equal')  
        st.pyplot(fig1)



def Knn__Manual_Input():
        if  filename :
            print("here")
            
            if e1 :
             data = {
                     'UserID': e1,
                     'No Of Abuse Report': e2,
                     'Rejected Friend Requests': e3,
                     'No Of Freind Requests Thar Are Not Accepted': e4,
                     'No Of Friends': e5,
                     'No Of Followers': e6,
                     'No Of Likes To Unknown Account': e7,
                     'No Of Comments Per Day': e8
                   }
             inputframe = pd.DataFrame([data])

             
            inputframe = inputframe[['UserID', 'No Of Abuse Report','No Of Freind Requests Thar Are Not Accepted','No Of Friends','No Of Followers','No Of Likes To Unknown Account','No Of Comments Per Day']]
            print(inputframe.loc[0])
            
            
            df=pd.read_csv(filename) 
            msk = np.random.rand(len(df)) < 0.7
            train = df[msk]
            test = inputframe
            testing_data=test.values[:, 0:7]
            features = train.values[:, 0:7]
            labels   = train.values[:, 8].astype('int')
            model3 = KNeighborsClassifier(n_neighbors=3)
            model3.fit(features,labels)
            predictions_model3 = model3.predict(testing_data)
            print('3.K-Nearest Neighbors  :\n')
            print('\n Predicted Class :',predictions_model3[0])
            show_predicted_label(predictions_model3[0])

def show_predicted_label(label):
        if label == 1:
            st.info('This twitter account is Fake', icon="ℹ️")
        else :
            st.info('This twitter account is Real', icon="ℹ️")


if st.button('Run Knn Algorithm'):
     Knn()

st.header('Manual input')
          

e1=st.text_input('Enter UserId')
e2=st.text_input('Enter No of Abuse Report')
e3=st.text_input('Enter No of Rejected Friend Requests')
e4=st.text_input('Enter No of Friend Requests That are not Accepted')
e5=st.text_input('Enter No of Friends')
e6=st.text_input('Enter No of followers')
e7=st.text_input('Enter No of likes To Unknown Account')
e8=st.text_input('Enter No of Comments Per Day')

if st.button('Predict'):
    Knn__Manual_Input()