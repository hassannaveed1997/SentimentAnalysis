# SentimentAnalysis
Sentiment analysis on text messages



STEP 0) Preparation
Download all the files provided.

If you have downloaded the pickle files, you can skip straight to step3. If the files need to be generated project.py will result in an error, you will need to follow step one and two.

Ensure the following libraries are installed/accessible:
nltk
pandas
sklearn
pickle
csv
numpy
os
re
tkinter

STEP 1) Preprocessing:

Run textProcessing.py from the command line using:

  python3 textProcessing.py
  
Make sure that textProcessing and the sentiment140 training csv file (from https://www.kaggle.com/kazanova/sentiment140) are in the same folder.

The script will then output a file called cleaned_data.csv in the same folder

STEP 2) Model Training:

Ensuring you have cleaned_data.csv from the previous step in the same folder, do the following:
- Run Logistic.py from the command line using:
  python3 Logistic.py
- Run SVM.py from the command line using:
  python3 SVM.py  
- Run NB.py from the command line using:
  python3 NB.py

All of these use a count vectorizor with unigrams. Now you should have 3 distinct pickle files in the pickled_models folder.

STEP 3) Running the classifier

Ensuring you have the 3 pickle files in the pickled_models folder, do the following:

- Run project.py from the command line using:

  python3 project.py
  
-This will open up a GUI. On the left most side, there is a send message button. In the middle there is a textbox for entering text.
On the right is a drop down menu of models. 

-Select the model from the drop down menu, enter the text and press the enter key (or click on the send message button). 

-If the sentiment is predicted to be positive, the text box will turn blue. If negative, it will turn red

-For Logistic and Naive Bayes models, the probability will also be printed in the terminal. (SVM only classifies)

