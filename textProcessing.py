def removeStopWords(text):
	#English stop words supported by the nltk library
	stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
	#split creates a list of words
	words = text.split()

	#For each word in the list check if it is a stop word
	for w in words:
		#if it is a stop word remove it from the list
		if w in stopWords:
			words.remove(w)
	return words


#p = "Nick likes to play football, however he is not too fond of tennis."
#print(removeStopWords(p))



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

p = "Nick likes to play football, however he is not too fond of tennis."
print(removeStopWords(p))
