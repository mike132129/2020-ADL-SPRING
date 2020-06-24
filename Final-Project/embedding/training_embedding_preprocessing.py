import json
import glob
import pdb
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm.auto import tqdm
import random
import numpy as np
from utils import normalize_text
import pickle

def load_data():

	# list data file
	# file_list = glob.glob('data/*')
	# file_list = file_list[:1]

	texts = []
	with open('./data/data.pkl', 'rb') as file:
		all_data = pickle.load(file)

	for data in all_data:
		texts.append(data['text'])


	# for file in file_list:
	# 	with open(file, 'r') as f:
	# 		doc_word = []
	# 		data = json.load(f)

	# 	for page in range(10):
	# 		try:
	# 			page_word = data['text'][page]['content'].split('\n')
	# 		except:
	# 			pass
	# 		doc_word += page_word

	# 	text = ''
	# 	for word in doc_word:

	# 		if len(text) + len(word) > 512:
	# 			texts.append(text)
	# 			text = ''

	# 		else:
	# 			text += word

	# 	texts.append(word)

	


	return texts

def create_mask(input_id):

    return [1 if idx != 0 else 0 for idx in input_id]

def tokenization(data, tokenizer):

	context = []
	
	for i in (data):

		text = normalize_text(i)
		encode = tokenizer.encode(text)
		context.append(encode)

	return context

def mask_label(context, tokenizer):

	labels = []

	for text in context:

		# Randomly choose 15% word to be masked
		label = [0]*len(text)
		mask_indices = random.sample(range(1, len(text)), k=int(len(text)*0.15))
		for mask_index in mask_indices:
			label[mask_index] = text[mask_index]
			text[mask_index] = 4

		labels.append(label)


	return context, np.array(labels)

def padding(context, labels):

	mask_att = []

	input_ids = pad_sequences(context, maxlen=512, dtype='long', padding='post', truncating='post')
	labels = pad_sequences(labels, value=0, maxlen=512, dtype='long', padding='post', truncating='post')
	for input_id in input_ids:
		mask_att += [create_mask(input_id)]

	return input_ids, np.array(mask_att), labels

def main():

	model_version = 'cl-tohoku/bert-base-japanese'
	tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
	data = load_data()
	context = tokenization(data, tokenizer)
	context, label = mask_label(context, tokenizer)
	input_ids, mask_attention, label = padding(context, label)
	np.save('./preprocess_data/input_ids.npy', input_ids)
	np.save('./preprocess_data/mask_attention.npy', mask_attention)
	np.save('./preprocess_data/label.npy', label)


if __name__ == '__main__':
	main()