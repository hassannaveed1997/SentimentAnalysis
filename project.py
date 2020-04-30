from tkinter import *
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

#reads in the vectorized features and the logitic regression model
(cv,lr)  = pickle.load(open('LR_model.pickle', 'rb')) 

#All the information to train the model can be inserted here.
def trainModel():

	return


#Takes in the word and an integer reprenting which model to use.
#Returns the prediction of 1 (positive) or -1 (negative) depending on the sentiment
def analyzeSentiment(word,model):
	print('sentiment analyszed on \"' + word + "\" and model is: " + model)
	pos_prob = 0 #probability of being positive

	if model == 'Logistic':

		print('opened logistic regression')
		input_text = [word] #for some reason count vectorization requires a list
		input_text_vect = cv.transform(input_text)
		pos_prob = lr.predict_proba(input_text_vect)[0][1]
		print('probability of positive is: ' + str(pos_prob))


	elif model == 'SVM':
		pass

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
	option = OptionMenu(gui, var, "Logistic", "SVM") #add more models if needed
	option.pack(side = LEFT)

	gui.mainloop()
	

