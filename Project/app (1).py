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

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR//tesseract.exe"

app = Flask(__name__)


MAX_SEQUENCE_LENGTH = 400
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50


word2vec = {}

# Glove file path
with open(os.path.join('C:/Users/HP/Desktop/webapp/glove.6B.%sd.txt' %EMBEDDING_DIM),encoding='utf-8') as f:
    # is just a space-seperated text file in the format:
    # word vec[0] vec[1] vec[2]....
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec


# tokenizer file path
with open('C:/Users/HP/Desktop/webapp/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def predict_label(data):
	# model file path 
	model = load_model('C:/Users/HP/Desktop/webapp/final_lstm.h5')

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
		correct_ans_img = request.files['que_image']
		ques = request.form['text']
		weight = request.form['int']
		weight = int(weight)


		img_path = "static/" + img.filename	
		img.save(img_path)

		corr_ans_img_path = "static/" + correct_ans_img.filename	
		correct_ans_img.save(corr_ans_img_path)


		image = Image.open(img_path)
		sentenc = pytesseract.image_to_string(image) 
		if len(sentenc) < 1:
			sentenc = "Text Unrecognized"
		print(len(sentenc))
		sentence  = [sentenc]

		corr_ans_image = Image.open(corr_ans_img_path)
		corr_ans_txt = pytesseract.image_to_string(corr_ans_image)
		if len(corr_ans_txt) < 1:
			corr_ans_txt = "Text Unrecognized"
		print(len(corr_ans_txt))
		#question  = [que]

		sequences = tokenizer.texts_to_sequences(sentence)
		# get word -> integer mapping
		word2idx = tokenizer.word_index

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

		marks = predict_label(data)
		marks = np.round(marks[0][0])	

		similarity_marks = get_document_similarity (corr_ans_txt, sentenc)
		print(similarity_marks)
		
	return render_template("index.html", question = corr_ans_txt, answer = sentenc,text_ques = ques, mark = marks, sim_marks = (similarity_marks*weight),mark_weight = weight) 


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
