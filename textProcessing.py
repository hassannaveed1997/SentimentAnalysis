#######################################################################
###################### USING NLTK LIBRARY #############################
#######################################################################
import pandas as pd
import textProcessing_methods as preprocess

#read data from .csv file
dataset = pd.read_csv(r'training.1600000.processed.noemoticon.csv',encoding = "ISO-8859-1")

# create data frame with only the sentiment label column and the text
# sentiment labels: 0: Negative, 2: Neutral, 4: Positive. 
df = pd.DataFrame(dataset)
cols = [0,5]
df = df[df.columns[cols]]

df.columns = ["Sentiment", "Tweet"]


#The following part is only intended to change the input format for the logistic regression
#It uses the above methods for processing the data, then saves it.
#The resulting dataframe contains two columns:
#1: Sentiment: with values 0 (negative) and 1 (positive)
#2: Tweets: Stemmized and cleaned version of tweets (note: not tokenized)

#temporarily using one fraction of the cleaned data in the LR model to save time. 
#replace dftemp with df in actual execution (can take upto one hour)
dftemp = df.sample(frac = 0.25, random_state = 50)

sentiment = dftemp['Sentiment']
tweet = dftemp['Tweet']

#replace all the 4's for positive to 1's
sentiment = sentiment.replace(4,1)
tweet_cleaned = []

#change from dataframe to list for processions
tweet = tweet.tolist() 

#processed the text
for i in range(len(tweet)):
    tweet_cleaned.append(preprocess.processText(tweet[i]))

#rejoins and untokenizes it
for j in range (len(tweet_cleaned)):
    tweet_cleaned[j] = ' '.join(tweet_cleaned[j])

#creates pandadataframe and saves it.
df = pd.DataFrame(list(zip(sentiment, tweet_cleaned)), 
               columns =['Sentiment', 'Tweet']) 
df.to_csv(r'cleaned_data.csv',header = True)