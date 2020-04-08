from makedata import *
import re
import numpy as np
from sklearn.utils import class_weight
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Model, Input, load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model, Input, load_model
from tensorflow.python.keras.layers.core import Reshape
from module import biLSTM_baseline, seq_2_seq
from module import seq_2_seq_att_LSTM, seq_2_seq_biLSTM_att
import pdb
from tqdm import tqdm
import json
from argparse import ArgumentParser
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--predict')
parser.add_argument('--load_model')
parser.add_argument('--output_path')
parser.add_argument('--valid_test')
args = parser.parse_args()

print('LOADING DATA')

with open('./data/X_tokenizer.pkl', 'rb') as file:
    X_tokenizer = pickle.load(file)
    file.close()

with open('./data/Y_tokenizer.pkl', 'rb') as file:
    Y_tokenizer = pickle.load(file)
    file.close()

X_embedding = np.load('./data/text_embedding.npy')

def text_to_one_list(X):
    for i in tqdm(range(len(X))):
        b = []
        for j in range(len(X[i])):
            b += X[i][j]
        X[i] = b
    return X

def process_test_data(test_data_path, tokenizer):

    print('DATA PREPROCESSING...')
    test_id = []
    test_text = []
    test_bound = []
    with open(test_data_path) as j:
        for each in j:
            each = json.loads(each)
            test_id += [each['id']]
            test_text += [each['text']]
            test_bound += [each['sent_bounds']]
    
    X_test = text_parsing(test_text, test_bound)
    X_test_seq = text_cleaning(X_test, [])
    
    test_sentence_length = []
    X_test_seq = text_to_sequence(X_test, test_sentence_length, tokenizer)
    X_test = text_to_one_list(X_test)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN, truncating='post', padding='post')
    return test_id, X_test, test_sentence_length

MAX_LEN = 100
num_words = 20000
EMBEDDING_DIM = 300
MAX_SUMMARY_LEN = 30

if args.predict == 'early' or args.predict == 'extractive':
    
    

    test_id, X_test, test_sentence_length = process_test_data(test_data_path=args.test_data_path, tokenizer=X_tokenizer)
    model = biLSTM_baseline(embedding_matrix=X_embedding, MAX_LEN=MAX_LEN, num_words=num_words, EMBEDDING_DIM=EMBEDDING_DIM,
                            LSTM_units=256, LSTM_dropout=0.5)

    # Load model
    print('LOADING MODEL')
    model.load_weights(args.load_model)
    # Predict result
    def predic(X, sentence_length):
        print('Start predicting..')
        with open(args.output_path, 'w') as file:
            pred = model.predict(X)
            pred = np.squeeze(pred)
            print('Post processing')
            for i in tqdm(range(len(X))):
                result = {}
                result["id"] = test_id[i]
                score = np.array([])
                start = 0
                for j in range(len(sentence_length[i])):
                    if not sentence_length[i]:
                        continue
                    mean = np.mean(pred[i][start:start + int(sentence_length[i][j])])
                    if np.isnan(mean):
                        continue
                    start += int(sentence_length[i][j])
                    score = np.append(score, mean)

                if score.size > 0:
                    if score.size > 2:
                        win = score.argsort()[-2:][::-1]
                        
                    if score.size == 1 or score.size == 2:
                        win = [score.argmax()]
                else:
                     win = [0]

                result["predict_sentence_index"] = [int(i) for i in win]
                file.write(json.dumps(result))
                file.write("\n")
                result.clear()
            file.close()

    predic(X_test, test_sentence_length)

if args.predict == 'abstractive_without_att':

    test_id, X_test, _ = process_test_data(test_data_path=args.test_data_path, tokenizer=X_tokenizer)
    # Prepare dictionary
    reverse_source_word_index = X_tokenizer.index_word
    reverse_target_word_index = Y_tokenizer.index_word
    target_word_index = Y_tokenizer.word_index

    model = load_model(args.load_model)
    encoder_inputs = model.input[0] # input-1
    encoder_embedding = model.layers[2]
    encoder_embedding_output = encoder_embedding(encoder_inputs)
    encoder_lstm = model.layers[4]
    encoder_outputs, state_h_encoder, state_c_encoder = encoder_lstm(encoder_embedding_output)
    encoder_states = [encoder_outputs, state_h_encoder, state_c_encoder]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(None,), name='input_3')
    decoder_state_input_c = Input(shape=(None,), name='input_4')
    decoder_hidden_state_input = Input(shape=(MAX_LEN, None))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding = model.layers[3]
    decoder_embedding_output = decoder_embedding(decoder_inputs)
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding_output, initial_state=decoder_states_inputs)
    decoder_states = [decoder_outputs, state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, 
                            decoder_state_input_c], [decoder_outputs] + [state_h_dec, state_c_dec])
    print(model.summary())


    def decode_sequence(X, MAX_SUMMARY_LEN):
        
        predicted_index = np.zeros((len(X), MAX_SUMMARY_LEN))
        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(X)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((len(X),1))
        target_seq[:, 0] = target_word_index['bos']

        for i in tqdm(range(MAX_SUMMARY_LEN)):
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            output_tokens = np.squeeze(output_tokens)
            sample_token_index = np.argmax(output_tokens, axis=1)

            predicted_index[:, i] = sample_token_index
            # Update the target sequence
            target_seq = sample_token_index.reshape(-1, 1)

            # Update the interal states
            e_h, e_c = h, c

        decoded_sentence = []
        for i in tqdm(range(len(predicted_index))):
            sentence = ''
            for j in range(MAX_SUMMARY_LEN):
                if predicted_index[i][j] == 0: # PAD
                    predicted_index[i][j] = 1 # END

                sampled_token = reverse_target_word_index[predicted_index[i][j]]
                if sampled_token == 'eos':
                    break
                else:
                    sentence += sampled_token + ' '

            decoded_sentence.append(sentence)

        return decoded_sentence

    def predic(X, MAX_SUMMARY_LEN):
        print('Start predicting')
        with open(args.output_path, 'w') as file:
            pred_sentence = decode_sequence(X, MAX_SUMMARY_LEN)
            for i in tqdm(range(len(X))):
                # print('predict the {}th text'.format(i))
                result = {}
                result["id"] = test_id[i]
                result["predict"] = pred_sentence[i]
                file.write(json.dumps(result))
                file.write("\n")
                result.clear()
            file.close()

    predic(X_test, MAX_SUMMARY_LEN)

if args.predict == 'abstractive_with_att':

    from tensorflow.python.keras.layers import Add, dot, Activation, concatenate, TimeDistributed, Dense
    test_id, X_test, _ = process_test_data(test_data_path=args.test_data_path, tokenizer=X_tokenizer)

    reverse_source_word_index = X_tokenizer.index_word
    reverse_target_word_index = Y_tokenizer.index_word
    target_word_index = Y_tokenizer.word_index

    Hidden_units = 250

    model = load_model(args.load_model)
    print(model.summary())

    encoder_inputs = model.input[0] # input-1
    encoder_embedding = model.layers[1]
    encoder_bilstm = model.layers[3]
    enc_concat_1 = model.layers[5]
    enc_concat_2 = model.layers[6]
    
    encoder_embedding_output = encoder_embedding(encoder_inputs)
    encoder_outputs, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder_bilstm(encoder_embedding_output)
    state_h_encoder = enc_concat_1([enc_forward_h, enc_backward_h])
    state_c_encoder = enc_concat_2([enc_forward_c, enc_backward_c])
    encoder_states = [encoder_outputs, state_h_encoder, state_c_encoder]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(2*Hidden_units,), name='input_3')
    decoder_state_input_c = Input(shape=(2*Hidden_units,), name='input_4')
    encoder_out = Input(shape=(MAX_LEN, 2*Hidden_units), name='encoder_output')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding = model.layers[4]
    decoder_embedding_output = decoder_embedding(decoder_inputs)
    decoder_lstm = model.layers[7]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embedding_output, initial_state=decoder_states_inputs)

    # # attention inference
    attention = model.layers[8] # Dot
    attention_wgt_layer = model.layers[9]
    context_layer = model.layers[10]
    concatenate_layer = model.layers[11]
    Dense1 = model.layers[12]
    Dense2 = model.layers[13]

    attention_output = attention([decoder_outputs, encoder_out])
    attention_weight = attention_wgt_layer(attention_output)
    context = context_layer([attention_weight, encoder_out])

    decoder_combined_context = concatenate_layer([context, decoder_outputs])
    output = Dense1(decoder_combined_context)
    output = Dense2(output)

    decoder_model = Model([decoder_inputs] + [encoder_out, decoder_state_input_h, decoder_state_input_c], 
                        [output] + [state_h_dec, state_c_dec])
    

    def decode_sequence(X, MAX_SUMMARY_LEN):

        e_out, e_h, e_c = encoder_model.predict(X)
        target_seq = np.zeros((len(X), 1))
        target_seq[:, 0] = target_word_index['bos']
        predicted_index = np.zeros((len(X), MAX_SUMMARY_LEN))

        for i in tqdm(range(MAX_SUMMARY_LEN)):
            dec_output, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            dec_ind = np.argmax(dec_output, axis=-1)
            target_seq = dec_ind.reshape(-1, 1)
            predicted_index[:, i] = dec_ind.reshape(1, -1)
            e_h, e_c = h, c


        decoded_sentence = []
        for i in tqdm(range(len(predicted_index))):
            sentence = ''
            for j in range(len(predicted_index[0])):
                if predicted_index[i][j] == 0:
                    predicted_index[i][j] = 1

                sampled_token = reverse_target_word_index[predicted_index[i][j]]
                if sampled_token == 'eos':
                    break
                else:
                    sentence += sampled_token + ' '

            decoded_sentence.append(sentence)
        return decoded_sentence

    def predic(X, MAX_SUMMARY_LEN):
        print('Start predicting')
        with open(args.output_path, 'w') as file:
            pred_sentence = decode_sequence(X, MAX_SUMMARY_LEN)
            for i in tqdm(range(len(X))):
                # print('predict the {}th text'.format(i))
                result = {}
                result["id"] = test_id[i]
                result["predict"] = pred_sentence[i]
                file.write(json.dumps(result))
                file.write("\n")
                result.clear()
            file.close()


    predic(X_test, MAX_SUMMARY_LEN)










































