#######################################################################
###################### USING NLTK LIBRARY #############################
#######################################################################
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

dataset = pd.read_csv(r'training.1600000.processed.noemoticon.csv')

# This method takes a 
def processText(text):

    textList = text.split()

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
    cleaners = ["@", "#", "www.", "https://", ".edu", ".com"]

    #check each subtring if contained in word
    for c in cleaners:
        #if word contains substring return True
        if c in word:
            return True

    #else return False
    return False
#TEST
p = " @mahad this is not the typical mel www.pornhub.com brooks film it was much less slapstick than most of his movies and actually had a plot that was followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moments that could have been fleshed out a bit more and some scenes that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting was good overall brooks himself did a good job without his characteristic speaking to directly to the audience again warren was the best actor in the movie but fume and sailor both played their parts well"
print(processText(p))
