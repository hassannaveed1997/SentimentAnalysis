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

#Save SVM model
if not os.path.exists("pickled_models"):
    os.makedirs("pickled_models")
    
save_svm = open("pickled_models/svm_model.pickle", "wb")
pickle.dump(svm, save_svm)
save_svm.close()


#Get features
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), svm.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)