from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

#importing data
data = pd.read_csv('cleaned_data.csv')
data.columns = ['Sentiment','Tweet']

train=data.sample(frac=0.8,random_state=50) #random state is a seed value
test=data.drop(train.index)

# assigning data to respective lists
Train_text = train['Tweet']
Train_values = train['Sentiment']
Test_text = test['Tweet']
Test_values = test['Sentiment']

#CV with ngram range 1,2
cv = CountVectorizer(binary=True, ngram_range= (1,2))
cv.fit(Train_text.values.astype('U'))
Train_text_vect = cv.transform(Train_text.values.astype('U'))
Test_text_vect = cv.transform(Test_text.values.astype('U'))

#TF-IDF with min doc frequency 2
# tfidf = TfidfVectorizer(ngram_range = (1,3),min_df = 2)
# tfidf.fit(Train_text.values.astype('U'))
# Train_text_vect = tfidf.transform(Train_text.values.astype('U'))
# Test_text_vect = tfidf.transform(Test_text.values.astype('U'))

#SVM model
svm = LinearSVC(C=0.01)
svm.fit(Train_text_vect, Train_values)
score = accuracy_score(Test_values, svm.predict(Test_text_vect))

print(score)


#Now we will extract the features for analysis
#create a dataframe containing coefficients from the regression and features from the count vectorizor
features = pd.DataFrame(cv.get_feature_names(), svm.coef_.tolist()).reset_index()

#rename the columns
features.columns = ['coef','feature']

#sort the features (most negative first)
features = features.sort_values(by=['coef'])

#print the 5 most positive and negative features
print(features.head(5))
print(features.tail(5))


# Save SVM model
if not os.path.exists("pickled_models"):
	os.makedirs("pickled_models")
	
pickle.dump((cv,svm), open('pickled_models/svm_model.pickle', 'wb'))