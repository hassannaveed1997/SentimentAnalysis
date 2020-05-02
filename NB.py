from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Multinomial Naive Bayes Classifier
# This file implements a MultinomiaNB Classifier and reports results based
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

#MultinomialNB that outputs classification report
nb = MultinomialNB()
nb.fit(Train_text_vect, Train_values)
score = accuracy_score(Test_values, nb.predict(Test_text_vect))

print(score)

# Create a dataframe containing coefficients from the regression 
# and features from the count vectorizor for analysis.

features = pd.DataFrame(vectorizer.get_feature_names(), nb.coef_.tolist()).reset_index()

#rename the columns
features.columns = ['coef','feature']

#sort the features (most negative first)
features = features.sort_values(by=['coef'])

# Print the 5 most positive and negative features
#print(features.head(5))
#print(features.tail(5))


# Save vectorizer and nb model to a pickle file in the models folder
if not os.path.exists("pickled_models"):
	os.makedirs("pickled_models")

pickle.dump((vectorizer,nb), open('pickled_models/nb_model.pickle', 'wb'))