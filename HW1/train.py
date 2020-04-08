from makedata import *
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.preprocessing.text import Tokenizer

import pdb
from tqdm import tqdm
import pandas as pd
import json
from module import biLSTM_baseline, seq_2_seq
from module import seq_2_seq_att_LSTM, seq_2_seq_biLSTM_att, seq_2_seq_biLSTM_att_weightDecay
from argparse import ArgumentParser

import pickle

# Load pretrained embedding

parser = ArgumentParser()
parser.add_argument('--train')
args = parser.parse_args()

print('LOADING.....')

X_train = np.load('./data/train.npy')
Y_train = np.load('./data/train_label.npy')
X_embedding = np.load('./data/text_embedding.npy')

# load tokenizer
with open('./data/X_tokenizer.pkl', 'rb') as file:
	X_tokenizer = pickle.load(file)

MAX_LEN = 100
num_words = 20000
EMBEDDING_DIM = 300

if args.train == 'early' or args.train == 'extractive':
	
	print('Constructing training model')
	model = biLSTM_baseline(embedding_matrix=embedding_matrix, MAX_LEN=MAX_LEN, num_words=num_words, EMBEDDING_DIM=EMBEDDING_DIM,
							LSTM_units=256, LSTM_dropout=0.5)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print(model.summary())

	# Deal with Imbalanced data
	# Collect train label
	collect_train_label = Y_train.reshape(1, -1)
	collect_train_label = np.squeeze(collect_train_label)
	class_weights_vec = class_weight.compute_class_weight('balanced', np.unique(collect_train_label), collect_train_label)

	pdb.set_trace()
	# Train
	print('Starting Training')
	model_file = "./model/extractive-save-model-{epoch:02d}.hdf5"
	checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]
	history = model.fit(X_train, Y_train.reshape(Y_train.shape[0], MAX_LEN, -1), batch_size=256, epochs=200, validation_split=0.1, verbose=1, callbacks=callbacks_list, shuffle=True, class_weight=class_weights_vec)


if args.train == 'abstractive_without_att':
	
	print('Constructing training model')
	model = seq_2_seq(X_embedding=X_embedding, MAX_LEN=MAX_LEN,
						num_words=num_words, EMBEDDING_DIM=EMBEDDING_DIM, 
						LSTM_units=150, LSTM_dropout=0.5)
	print(model.summary())

	# use sparse categorical crossentropy since it convert the integer sequence to a one-hot vector on the fly
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	print('Starting Training')
	model_file = "./model/seq2seq-save-model-{epoch:02d}.hdf5"
	checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, 
								save_best_only=False, mode='max', period=10)
	callbacks_list = [checkpoint]
	history = model.fit([X_train, Y_train[:,:-1]], Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)[:,1:],  batch_size=300, epochs=500, validation_split=0.1, verbose=1, callbacks=callbacks_list, shuffle=True)
	


if args.train == 'abstractive_with_att':

	print('Constructing training model')
	model = seq_2_seq_biLSTM_att(X_embedding=X_embedding, MAX_LEN=MAX_LEN,
											num_words=num_words, EMBEDDING_DIM=EMBEDDING_DIM, 
											LSTM_units=250, LSTM_dropout=0.5)
	
	print(model.summary())
	# Try different loss function [categorical_crossentropy]
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
	model_file = "./model/seq2seq-att-save-model-{epoch:02d}.hdf5"
	checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max', period=5)
	callbacks_list = [checkpoint]
	
	history = model.fit([X_train, Y_train[:,:-1]], Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)[:,1:],  batch_size=128, epochs=100, validation_split=0.1, verbose=1, callbacks=callbacks_list, shuffle=True)





















