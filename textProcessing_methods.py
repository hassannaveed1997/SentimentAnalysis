from nltk.tokenize import word_tokenize
from nltk.stem.snowball import *
#from nltk.corpus import stopwords

# processText takes in a string as it's parameter 
# removes stopwords such as usernames, email addresses,
# websites and hashtags and returns a tokenized and 
# stemmed list of words from the text.
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

    #For each word in the list check if it is a stop word using nltk function
    #resulted in depreciated performace so not being used
    #for w in words:
    #    if w in stopwords.words('english'):
    #        words.remove(w)


    #instance of a Snowball stemmer
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
    cleaners = ["@", "#", "www.", "Www.", "https://", "Https://" ,"http" ,".edu", ".com"]

    #check each subtring if contained in word
    for c in cleaners:
        #if word contains substring return True
        if c in word:
            return True

    #else return False
    return False
