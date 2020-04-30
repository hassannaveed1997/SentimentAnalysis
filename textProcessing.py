#######################################################################
###################### USING NLTK LIBRARY #############################
#######################################################################
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# This method takes a 
def processText(text):
	#tokenize by words
    words = word_tokenize(text)

    #For each word in the list check if it is a stop word
    for w in words:
    	#if it is a stop word remove it from the list
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

#TEST
p = "this is not the typical mel brooks film it was much less slapstick than most of his movies and actually had a plot that was followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moments that could have been fleshed out a bit more and some scenes that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting was good overall brooks himself did a good job without his characteristic speaking to directly to the audience again warren was the best actor in the movie but fume and sailor both played their parts well"
print(processText(p))
