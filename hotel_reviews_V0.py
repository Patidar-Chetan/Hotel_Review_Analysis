# Import the Libraries
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for Text
from pattern.en import lemma
from gensim.parsing.preprocessing import remove_stopwords
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Libraries for Model Validation and Test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(page_title='Hotel Reviews', page_icon='logo.png')


data1 = pd.read_csv('hotel_reviews.csv')
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()
if 'data_label' not in st.session_state:
    st.session_state['data_label'] = pd.DataFrame()
    
if 'start' not in st.session_state:
    st.session_state['start'] = 0
st.session_state['data'] = data1
st.session_state['data_label'] = data1

def model_validation():
    try:
        st.image('download.jpg', 'Hotel Reviews')
        st.header('**Model Prediction**')
        st.write('---')


        data1 = st.session_state['data_label']
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        otp_data = st.session_state['data']
        test_data = pd.DataFrame(otp_data['Review'])
        #st.write(test_data)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # Data Preprocessing --------------------------------------------------------------------------------------------------------------------------------------


        def lemmatized(text):
            try:
                return " ".join([lemma(wd) for wd in text.split()])
            except RuntimeError:
                return " ".join([lemma(wd) for wd in text.split()])

        pos = [5,4]
        neg = [1,2]
        neu=[3]

        def sentiment(rating):
            if rating in pos:
                return "positive"
            elif rating in neg:
                return "negative"
            elif rating in neu:
                return "neutral"

        def clean(data, review_col_name, rating_col_name):
            
            data1 = data.copy()
            #st.write('1', data1)
            # Word Length of reviews
            data1['word_len_review'] = data1[review_col_name].apply(lambda x: len(x.split()))
            #st.write('2', data1)
            # String Length of reviews
            data1['string_len_review'] = data1[review_col_name].apply(lambda x: len(x))
            #st.write('3', data1)
            
            # Lowercase the reviews
            data1['cleaned']=data1['Review'].apply(lambda x: x.lower())
            # Remove digits and punctuation marks
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub('[^a-z]',' ', x))
            # Removing extra spaces if present
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub(' +',' ',x))
            #st.write('4', data1)
            data1['cleaned']=data1['cleaned'].apply(lambda x: remove_stopwords(x))
            #st.write('5', data1)
            data1['lemmatization']=data1['cleaned'].apply(lambda x: lemmatized(x))
            
            #st.write('6', data1)
            data1['Sentiment'] = data1[rating_col_name].apply(sentiment)
            data1['label'] = data1['Sentiment'].map({'positive':1, 'negative':-1, 'neutral':0})
            #st.write('7', data1)
            
            return data1

        data1 = clean(data1, 'Review', 'Rating')
        st.subheader('Training DataSet')
        st.write(data1)
        def clean(data, review_col_name):
            
            data1 = data.copy()
            #st.write('1', data1)
            # Word Length of reviews
            data1['word_len_review'] = data1[review_col_name].apply(lambda x: len(x.split()))
            #st.write('2', data1)
            # String Length of reviews
            data1['string_len_review'] = data1[review_col_name].apply(lambda x: len(x))
            #st.write('3', data1)
            
            # Lowercase the reviews
            data1['cleaned']=data1['Review'].apply(lambda x: x.lower())
            # Remove digits and punctuation marks
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub('[^a-z]',' ', x))
            # Removing extra spaces if present
            data1['cleaned']=data1['cleaned'].apply(lambda x: re.sub(' +',' ',x))
            #st.write('4', data1)
            data1['cleaned']=data1['cleaned'].apply(lambda x: remove_stopwords(x))
            #st.write('5', data1)
            data1['lemmatization']=data1['cleaned'].apply(lambda x: lemmatized(x))
            
            return data1
        st.session_state['data_label'] = data1

        ###########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        test_data1 = clean(test_data, 'Review')
        st.subheader('Testing DataSet')
        st.write(test_data1)


        # Convert Text Data Into Numerical Data
        from sklearn.feature_extraction.text import TfidfVectorizer
        tv = TfidfVectorizer(max_features=5000)
        X = tv.fit_transform(data1['lemmatization']).toarray()
        X_test = tv.transform(test_data1['lemmatization']).toarray()
        X=pd.DataFrame(X, columns=tv.get_feature_names())#.set_flags(allows_duplicate_labels=True)
        #st.write('1.final_data', X.shape)
        X_test=pd.DataFrame(X_test, columns=tv.get_feature_names())
        #st.write('1.final_test_data', X_test.shape)
        #final_data = pd.concat([data1[['label','word_len_review','string_len_review']],X], axis=1)
        #st.write('1.final_data')

        from sklearn.preprocessing import MinMaxScaler
        minmax_model = MinMaxScaler(feature_range = (0 , 1))

        minmax = minmax_model.fit_transform(data1[['word_len_review','string_len_review']])
        test_minmax = minmax_model.transform(test_data1[['word_len_review','string_len_review']])
        minmax = pd.DataFrame(minmax, columns=['word_len_review','string_len_review'])
        test_minmax = pd.DataFrame(test_minmax, columns=['word_len_review','string_len_review'])

        final_data = pd.concat([data1['label'],minmax,X], axis=1)
        #st.write('2.final_data', final_data.shape)
        final_test_data = pd.concat([test_minmax,X_test], axis=1)
        #st.write('2.final_test_data', final_test_data.shape)


        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        model = SelectKBest(score_func=chi2, k='all')
        fit = model.fit(final_data.iloc[:,1:], final_data.iloc[:,0])
        scores = np.around(fit.scores_, 3)

        id_cols = list(np.where(scores>0.5)[0])
        idx_cols = [x+1 for x in id_cols]

        final_data = pd.concat([final_data.iloc[:,0],final_data.iloc[:,idx_cols]], axis=1)
        #st.write('3.final_data', final_data.shape)
        final_test_data = final_test_data.iloc[:,id_cols]
        #st.write('3.final_test_data', final_test_data.shape)

        # Spliting into X, y
        X = final_data.iloc[:,1:]
        y = final_data.iloc[:,0]

        X_test = final_test_data

        # Model Training
        #model = MultinomialNB()

        # Hyper parameter Value
        #kfold = KFold()
        #alpha = np.arange(0.1, 1.1, 0.1)
        #param_grid = {'alpha':alpha}

        # Hyper parameter tunning using GridSearchCV
        #model = MultinomialNB()
        #grid = GridSearchCV(estimator=model, param_grid=param_grid , cv = kfold, n_jobs=2)
        #grid.fit(X, y)
        #para = grid.best_params_
        #st.write(para)
        #model = MultinomialNB(alpha=para['alpha'])
        model = MultinomialNB(alpha=0.1)
        model.fit(X, y)
        #st.write('Model Fitted')


        y_train_pred = model.predict(X)

        st.subheader('Training Accuracy')
        st.write('Train Accuracy Score : ' , round(accuracy_score(y, y_train_pred),3))
        st.write('Train F1 Score : ' , round(f1_score(y, y_train_pred, average='weighted'),3))
        st.write('Train Precision Score : ' , round(precision_score(y, y_train_pred, average='weighted'),3))
        st.write('Train Recall Score : ' , round(recall_score(y, y_train_pred, average='weighted'),3))

        # print classification report
        st.text('Model Report on Training DataSet:\n ' 
                +classification_report(y, y_train_pred, digits=4))

        cm = confusion_matrix(y, y_train_pred)
        dt = {'Negative':list(cm[0]), 'Neutral':list(cm[1]), 'Positive':list(cm[2])}
        # Confusion Matrix for Train Data
        cm = pd.DataFrame(dt, index=['Negative', 'Neutral', 'Positive'])

        sns.set_theme(style='dark')
        sns.set(rc={'axes.facecolor':'#282828', 'figure.facecolor':'#282828'})

        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt='.0f', ax=ax)
        #ax.tick_params(grid_color='r', labelcolor='r', color='r')

        plt.ylabel('Predictions', fontsize=18)
        plt.xlabel('Actuals', fontsize=18)
        plt.title('Training Confusion Matrix', fontsize=18)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(colors='white')
        ax.figure.axes[-1].tick_params(colors='white')
        plt.show()
        st.write(fig)

        st.text("")
        st.text("")
        st.write('#')

        # Training Validation End___________________________________________________________________________________

                
        # Testing Validation________________________________________________________________________________________

        #Predict for X dataset       
        y_test_pred = model.predict(X_test)

        otp_data['label'] = y_test_pred
        otp_data['Sentiment'] = otp_data['label'].map({-1:'Negative', 0:'Neutral', 1:'Positive'})

        # Fuction define to color the dataframe
        def color_df(clas):
            if clas == 'Negative':
                color = 'tomato'
            elif clas == 'Positive':
                color = 'green'
            else:
                color = 'dimgrey'
                
            return f'background-color: {color}'

        # Final DataFrame
        st.subheader('Classified Data or Output')
        st.dataframe(otp_data.style.applymap(color_df, subset=['Sentiment']))

        # Value Counts of Final Dataframe
        dt = {'Sentiment_Classification':otp_data['Sentiment'].value_counts().index.tolist(), 
              'Counts':otp_data['Sentiment'].value_counts().values.tolist()}
        value_counts = pd.DataFrame(dt)
        st.subheader('Value Counts of Classified Data')
        st.dataframe(value_counts.style.applymap(color_df, subset=['Sentiment_Classification']))


        # Test Validation End_______________________________________________________________________________________
                
        # Model Validation End-------------------------------------------------------------------------------------------------------------------------------------------------------------
                
            
    except:
        st.error('Go to **Main Page** and Upload the DataSet')


model_validation()
    