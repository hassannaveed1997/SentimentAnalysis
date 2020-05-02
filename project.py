from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tkinter import *
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import textProcessing_methods
#reads in the vectorized features and the logitic regression model

(vectLR,lr)  = pickle.load(open('pickled_models/LR_model.pickle', 'rb')) 
(vectSVM, svm) = pickle.load(open('pickled_models/svm_model.pickle', 'rb')) 
(vectNB, nb) = pickle.load(open('pickled_models/nb_model.pickle', 'rb')) 


#process the text thsat is input in the model

#Takes in the text and an integer reprenting which model to use.
#Returns the prediction of 1 (positive) or -1 (negative) depending on the sentiment
def analyzeSentiment(text,model):
	pos_prob = 0 #probability of being positive
	
	#process the text to make it the same as the model
	text = textProcessing_methods.processText(text)
	
	#combine it back into a string instead of a list
	text = ' '.join(text)

	print('sentiment analyszed on \"' + text + "\" and model is: " + model)

	if model == 'Logistic':

		input_text = [text] #for some reason count vectorization requires a list

		#transform to a vector of features
		input_text_vect = vectLR.transform(input_text)

		#get the predicted probability. The result is a 2x1 matrix, we only need the positive probability
		pos_prob = lr.predict_proba(input_text_vect)[0][1]
		print('probability of positive is: ' + str(pos_prob))


	elif model == 'SVM':
		input_text = [text] #for some reason count vectorization requires a list

		#transform to a vector of features
		input_text_vect = vectSVM.transform(input_text)

		#get the predicted probability. The result is a 1x1 list, we only need the value
		result = svm.predict(input_text_vect)[0]
		print(result)

		#result is a boolean of 1 or 0. We change our predicted probability accordingly
		if result > 0.5:
			pos_prob = 1
		else:
			pos_prob = 0

	else: #naive bayes

		input_text = [text] #for some reason count vectorization requires a list

		#transform to a vector of features
		input_text_vect = vectNB.transform(input_text)

		#get the predicted probability. The result is a 2x1 matrix, we only need the positive probability
		pos_prob = nb.predict_proba(input_text_vect)[0][1]
		print('probability of positive is: ' + str(pos_prob))



	if pos_prob > 0.5:
		message.configure(background = 'sky blue')
	else:
		message.configure(background='indian red')
	
	return

#when the enter button is pressed in the text
def buttonPressed(event):
	analyzeSentiment(message.get(),var.get())
	return

#when the submit button is pressed
def buttonPressed2():
	analyzeSentiment(message.get(),var.get())
	return

if __name__ == "__main__": 
	gui = Tk()
	# set the background colour of GUI window 
	gui.configure(background='white') 

	# set the title of GUI window 
	gui.title("Sentimental Messaging Service (SMS)") 

	# set the configuration of GUI window 
	gui.geometry("700x50") 

	#create the submit button and place it properly
	submit = Button(gui, text="Send message", fg="Black", bg="grey", command = buttonPressed2) 
	submit.pack(side = LEFT)

	#create a message textfield. Pressing enter will act as the button is pressed.
	message = Entry(gui,width = 40)
	message.bind("<Return>",buttonPressed)
	message.pack(side = LEFT)

	#create option button
	var =StringVar(gui)
	var.set("Logistic") #default is logistic
	option = OptionMenu(gui, var, "Logistic", "SVM", "Naive Bayes") #add more models if needed
	option.pack(side = LEFT)

	gui.mainloop()
	

