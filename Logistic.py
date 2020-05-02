from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import csv
import numpy as np
import pandas as pd
import os
import re

#read in the dataset of cleaned tweets
data = pd.read_csv('cleaned_data.csv')

#randomly set 80% of the data for training and 20% for testing
train=data.sample(frac=0.8,random_state=50) 
test=data.drop(train.index)

#seperate the Tweet and Sentiment columns
train_text = train['Tweet']
train_values = train['Sentiment']
test_text = test['Tweet']
test_values = test['Sentiment']

#A count vectorizer creates a huge dataframe with each word in the corpus.
#Setting binary to True has the dataframe contain one or zero depending on whether or not a word appears in a tweet.
#Alternative(binary = False) or use TF-IDF vectorizor
cv = CountVectorizer(binary=True)

#fit the model on the existing training corpus (Note: text needs to encoded as unicode, so astype('U') is included)
cv.fit(train_text.values.astype('U'))

#creates the dataframes on the training and test data
train_text_vect = cv.transform(train_text.values.astype('U'))
test_text_vect = cv.transform(test_text.values.astype('U'))


c = 0.5 #hyperparameter value for the logistic regression

#fit and train the logistic regression
lr = LogisticRegression(C=c)
lr.fit(train_text_vect, train_values)

#predict and score
score = accuracy_score(test_values, lr.predict(test_text_vect))
print ("Accuracy Score for C ="+"  str(c) "+" equals:   " + str(score))
#for uncleaned data this is:
# Accuracy Score for C = 0.01   equals:   
# 0.789478125
# Accuracy Score for C = 0.05   equals:   
# 0.797440625
# Accuracy Score for C = 0.1   equals:   
# 0.799603125
# Accuracy Score for C = 0.25   equals:   
# 0.801228125
# Accuracy Score for C = 0.5   equals:   
# 0.801584375

#Now we will extract the features for analysis
#create a dataframe containing coefficients from the regression and features from the count vectorizor
features = pd.DataFrame(cv.get_feature_names(), lr.coef_.tolist()).reset_index()

#rename the columns
features.columns = ['coef','feature']

#sort the features (most negative first)
features = features.sort_values(by=['coef'])

#print the 5 most positive and negative features
print(features.head(5))
print(features.tail(5))


pickle.dump((cv,lr), open('LR_model.pickle', 'wb'))
#loaded_model = pickle.load(open(filename, 'rb'))


