#######################################################################
###################### USING NLTK LIBRARY #############################
#######################################################################
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

def processText(text):

    # make a list of words split with spaces
    textList = text.split()

    #convert all words to lower case
    for i in range(len(textList)):
        textList[i] = textList[i].lower()

    #remove usernames, email addresses, websites and hastags
    for t in textList:
        if (deepClean(t)):
            textList.remove(t)
    text = ' '.join(textList)

	#tokenize by words
    words = word_tokenize(text)

    #For each word in the list check if it is a stop word
    for w in words:
        if w in stopwords.words('english'):
            words.remove(w)

    #instance of a stemmer
    stemmer = SnowballStemmer("english")
    stemWords = []


    #iterate over all tokenized words to stem and store in list
    for t in words:
    	stemWords.append(stemmer.stem(t))

    	#return the list of tokenized stem words of the text passed to the method
    return stemWords


# This method takes a single string word checks it against
# a list of substrings to determine whether the word has 
# any useful information regarding the sentiment of the text.
# returns True if word is an email address, username, hashtag or 
# website else returns False.
def deepClean(word):

    # List of substrings to check if present in the word:
    # '@',".edu", ".com" may represent usernames and email addresses 
    # '#' represents hastags for social media 
    # 'www.', "https://", ".edu", ".com" may represent websites
    cleaners = ["@", "#", "www.", "Www.", "https://", "Https://" ,  ".edu", ".com"]

    #check each subtring if contained in word
    for c in cleaners:
        #if word contains substring return True
        if c in word:
            return True

    #else return False
    return False

#read data from .csv file
dataset = pd.read_csv(r'training.1600000.processed.noemoticon.csv',encoding = "ISO-8859-1")

# create data frame with only the sentiment label column and the text
# sentiment labels: 0: Negative, 2: Neutral, 4: Positive. 
df = pd.DataFrame(dataset)
cols = [0,5]
df = df[df.columns[cols]]

df.columns = ["Sentiment", "Tweet"]

# three lists to separate different sentiments
pos = []
neg = []
neutral = []

# the first 1279999 tweets (80%) are used to train the model.  
trainLen = int((len(df)*0.8))

#separate the three different sentiments into three different lists.
for i in range(trainLen):
    #if negative
    if (int(df.loc[i,"Sentiment"]) == 0):
        neg.append(str(df.loc[i,"Tweet"]))
    #if neutral
    elif (int(df.loc[i,"Sentiment"]) == 2):
        neutral.append(str(df.loc[i,"Tweet"]))
    #if positive
    else:
        pos.append(str(df.loc[i,"Tweet"]))

#three lists to contain processed texts of different sentiments
posList = []
negList = []
neutralList = []

#Process text for each sentiment and store in appropriate list
for j in range(len(pos)):
    posList.append(processText(pos[j]))

for j in range(len(neg)):
    negList.append(processText(neg[j]))

for j in range(len(neutral)):
    neutralList.append(processText(neutral[j]))

#Create a list for training (need to add corresponding labelList)
trainList = posList + negList + neutralList


#Hassan:
#The following part is only intended to change the input format for the logistic regression
#It uses the above methods for processing the data, then saves it.
#The resulting dataframe contains two columns:
#1: Sentiment: with values 0 (negative) and 1 (positive)
#2: Tweets: Stemmized and cleaned version of tweets (note: not tokenized)

#temporarily using one fraction of the cleaned data in the LR model to save time. 
#replace dftemp with df in actual execution (can take upto one hour)
dftemp = df.sample(frac = 0.02, random_state = 50)

sentiment = dftemp['Sentiment']
tweet = dftemp['Tweet']

#replace all the 4's for positive to 1's
sentiment = sentiment.replace(4,1)
tweet_cleaned = []

#change from dataframe to list for processions
tweet = tweet.tolist() 

#processed the text
for i in range(len(tweet)):
    tweet_cleaned.append(processText(tweet[i]))

#rejoins and untokenizes it
for j in range (len(tweet_cleaned)):
    tweet_cleaned[j] = ' '.join(tweet_cleaned[j])

#creates pandadataframe and saves it.
df = pd.DataFrame(list(zip(sentiment, tweet_cleaned)), 
               columns =['Sentiment', 'Tweet']) 
df.to_csv(r'cleaned_data.csv',header = True)
