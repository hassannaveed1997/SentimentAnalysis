#######################################################################
###################### USING NLTK LIBRARY #############################
#######################################################################
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd


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
    posList.append(processText(pos[i]))

for j in range(len(neg)):
    posList.append(processText(neg[i]))

for j in range(len(neg)):
    posList.append(processText(neg[i]))


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



