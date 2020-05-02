from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from sklearn.metrics import classification_report

# SVC Classifier
# This file implements a LinearSVC Classifier and reports results based
# on the processed Tweet dataset.

# Importing csv file into a dataframe
data = pd.read_csv('cleaned_data.csv')
data.columns = ['Number','Sentiment','Tweet']

# Train-test split with seed value 50
train=data.sample(frac=0.8,random_state=50)
test=data.drop(train.index)

# Assigning data to respective lists
Train_text = train['Tweet']
Train_values = train['Sentiment']
Test_text = test['Tweet']
Test_values = test['Sentiment']

# CV and TF-IDF Vectorizers (Uncomment code block to use)

# vectorizer = CountVectorizer(binary=True, ngram_range= (1,1))
# vectorizer.fit(Train_text.values.astype('U'))
# Train_text_vect = vectorizer.transform(Train_text.values.astype('U'))
# Test_text_vect = vectorizer.transform(Test_text.values.astype('U'))

vectorizer = TfidfVectorizer(ngram_range = (1,3),min_df = 2)
vectorizer.fit(Train_text.values.astype('U'))
Train_text_vect = vectorizer.transform(Train_text.values.astype('U'))
Test_text_vect = vectorizer.transform(Test_text.values.astype('U'))

#Linear SVC model that outputs classification report (change c value to investigate regularization parameter)
c = 0.25
svm = LinearSVC(C=c)
svm.fit(Train_text_vect, Train_values)
score = classification_report(Test_values, svm.predict(Test_text_vect))

print("For c value", + c)
print(score)

# Create a dataframe containing coefficients from the regression 
# and features from the count vectorizor for analysis.

features = pd.DataFrame(vectorizer.get_feature_names(), svm.coef_.tolist()).reset_index()

#rename the columns
features.columns = ['coef','feature']

#sort the features (most negative first)
features = features.sort_values(by=['coef'])

# Print the 5 most positive and negative features
#print(features.head(5))
#print(features.tail(5))


# Save vectorizer and SVM model to a pickle file in the models folder
if not os.path.exists("pickled_models"):
	os.makedirs("pickled_models")

pickle.dump((vectorizer,svm), open('pickled_models/svm_model.pickle', 'wb'))