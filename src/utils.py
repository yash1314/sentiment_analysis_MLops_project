import os
import sys
import joblib
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import precision_score


nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer() 
lemma = WordNetLemmatizer()

def save_obj(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        logging.info('Error occured in utils save_obj')
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
    X_test = tfidf_vectorizer.transform(X_test).toarray()

    try:
        report = {}
        for i in range(len(models)):

            model = list(models.values())[i]

            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            test_model_score = precision_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report, tfidf_vectorizer

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_obj in utils')
        raise CustomException(e, sys)
             

def nlp_transform(text):
    text = text.lower() # lowercasing
    text = nltk.word_tokenize(text) # splitting sentence into words
    
    y = []
    for i in text: # removing special characters
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text: # stop words and helping words removal.
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(lemma.lemmatize(i))
    
    return " ".join(y)