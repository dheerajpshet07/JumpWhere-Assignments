from flask import Flask, render_template, request
from keras.preprocessing import image
import pickle
import cv2
import numpy as np


import os
import pandas as pd

import numpy as np
import matplotlib as plt

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from document_similarity import get_document_similarity

import pickle
from keras.models import load_model

from PIL import Image
import pytesseract

app = Flask(__name__)


MAX_SEQUENCE_LENGTH = 400
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50


word2vec = {}

# Glove file path
with open(os.path.join('/glove.6B.%sd.txt' %EMBEDDING_DIM)) as f:
    # is just a space-seperated text file in the format:
    # word vec[0] vec[1] vec[2]....
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec


# tokenizer file path
with open('/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def predict_label(data):
	# model file path 
	model = load_model('/final_lstm.h5')

	marks = model.predict(data)

	return marks


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		que_keywords = request.form['text']

		img_path = "static/" + img.filename	
		img.save(img_path)

		image = Image.open(img_path)
		sentenc = pytesseract.image_to_string(image)
		sentence  = [sentenc]

		sequences = tokenizer.texts_to_sequences(sentence)
		# get word -> integer mapping
		word2idx = tokenizer.word_index

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

		marks = predict_label(data)
		marks = np.round(marks[0][0])	

		similarity_marks = get_document_similarity (que_keywords, sentenc)
		if similarity_marks > 0.5:
			similarity_marks = similarity_marks * 0.5
		else:
			similarity_marks = similarity_marks * 0.1

		total_marks = (marks*.5) + (similarity_marks*10)
		
		

	return render_template("index.html", question = sentenc, marks = np.round(total_marks) , img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
