import json
import re
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pdb
from tqdm import tqdm
import os
import pandas as pd
import pickle

def text_parsing(article, bound):
    parsing_sentence = []
    for i in tqdm(range(len(article))):
        temp = []
        for j in range(len(bound[i])):
            sentence = article[i][bound[i][j][0]:bound[i][j][1]]
            temp.append(sentence)
        parsing_sentence.append(temp)
    return parsing_sentence

def text_cleaner(text):
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                               "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                               "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                               "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                               "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                               "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                               "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                               "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                               "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                               "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                               "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                               "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                               "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                               "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                               "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                               "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                               "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                               "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                               "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                               "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                               "you're": "you are", "you've": "you have"}
    newString = text.lower()
    # newString = BeautifulSoup(newString, "lxml").text
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
 
    return newString

def text_cleaning(article, all_text):
    for i in tqdm(range(len(article))):
        for j in range(len(article[i])):
            article[i][j] = text_cleaner(article[i][j])
            # article[i][j] = correct_spellings(article[i][j])
            all_text.append(article[i][j])
    return article

def load_pretrained_embedding():
    print('Indexing word vectors.')

    GLOVE_DIR = './data'
    embedding_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return embedding_index

def prepare_embedding_matrix(num_words, EMBEDDING_DIM, word_index, embedding_index, MAX_NUM_WORDS):
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def text_to_sequence(X, length, tokenizer):
    for i in tqdm(range(len(X))):
        length_each = []
        for j in range(len(X[i])):
            sentence = np.array(tokenizer.texts_to_sequences([X[i][j]]))
            sentence = sentence.reshape(-1,)
            X[i][j] = sentence.tolist()
            length_each.append(len(X[i][j]))
        length.append(length_each)
    return X

def text_to_one_list(X):
    for i in tqdm(range(len(X))):
        b = []
        for j in range(len(X[i])):
            b += X[i][j]
        X[i] = b
    return X
# split
def load_data(target):

    all_article = []
    train_artic = []
    train_bound = []
    train_abstr = []
    train_extra = []

    # Load all data
    with open('./data/train.jsonl') as j:
        for each in j:
            each = json.loads(each)
            if each['text'] == "\n":
                # print(each['id'])
                continue
            train_artic += [each['text']]
            train_bound += [each['sent_bounds']]
            train_abstr += [each['summary']]
            train_extra += [each['extractive_summary']]
        j.close()

    valid_artic = []
    valid_bound = []
    valid_abstr = []
    valid_extra = []

    with open('./data/valid.jsonl') as j:
        for each in j:
            each = json.loads(each)
            if each['text'] == "\n":
                print(each['id'])
            valid_artic += [each['text']]
            valid_bound += [each['sent_bounds']]
            valid_abstr += [each['summary']]
            valid_extra += [each['extractive_summary']]
        j.close()

    test_artic = []
    test_bound = []

    with open('./data/test.jsonl') as j:
        for each in j:
            each = json.loads(each)
            if each['text'] == "\n":
                print('null article in test data:', each['id'])
            test_artic += [each['text']]
            test_bound += [each['sent_bounds']]
        j.close()

    if target == 'extractive':

        # Parsing article
        # Split each article into lists
        X_train = text_parsing(train_artic, train_bound)
        X_valid = text_parsing(valid_artic, valid_bound)
        X_test = text_parsing(test_artic, test_bound)

        # Data cleaning
        
        print('Data cleaning.')

        X_all_text = []
        X_train_seq = text_cleaning(X_train, X_all_text)
        X_valid_seq = text_cleaning(X_valid, [])
        X_test_seq = text_cleaning(X_test, [])

        # Load pretrained

        MAX_LEN = 100
        MAX_NUM_WORDS = 20000
        EMBEDDING_DIM = 300

        # finally, vectorize the text samples into a 2D integer tensor
        print('Tokenization')
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

        # Fit the tokenizer on our text
        tokenizer.fit_on_texts(X_all_text)

        # Get all words that the tokenizer knows
        word_index = tokenizer.word_index
        num_words = len(word_index) + 1
        print('Found %s unique tokens' % len(word_index))


        print('text to sequence vector')
        train_sentence_length = []
        valid_sentence_length = []
        test_sentence_length = []

        X_train_seq = text_to_sequence(X_train, train_sentence_length, tokenizer)
        X_valid_seq = text_to_sequence(X_valid, valid_sentence_length, tokenizer)
        X_test_seq = text_to_sequence(X_test, test_sentence_length, tokenizer)

        print('Making Label...')
        def make_label(X, extractive):
            label = []
            for i in range(len(X)):
                temp = []
                extrac_index = extractive[i]
                for j in range(len(X[i])):
                    if j == extrac_index:
                        temp += [1] * len(X[i][j])
                    else:
                        temp += [0] * len(X[i][j])

                label.append(temp)
            return label

        Y_train = make_label(X_train, train_extra)
        Y_valid = make_label(X_valid, valid_extra)

        X_train = text_to_one_list(X_train)
        X_valid = text_to_one_list(X_valid)
        X_test = text_to_one_list(X_test)

        X_train = pad_sequences(X_train, maxlen=MAX_LEN, truncating='post', padding='post') # Each sentence is padding to a size=max_len vector
        X_valid = pad_sequences(X_valid, maxlen=MAX_LEN, truncating='post', padding='post')
        X_test = pad_sequences(X_test, maxlen=MAX_LEN, truncating='post', padding='post')
        Y_train = pad_sequences(Y_train, maxlen=MAX_LEN, truncating='post', padding='post')
        Y_valid = pad_sequences(Y_valid, maxlen=MAX_LEN, truncating='post', padding='post')

        # Preparing embedding matrix
        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

        np.save('./data/extractive_label.npy', Y_train)
        return 

    if target == 'abstractive':

        # Parsing article into pieces of sentence
        X_train = text_parsing(train_artic, train_bound)
        X_valid = text_parsing(valid_artic, valid_bound)
        X_test = text_parsing(test_artic, test_bound)

        # Data cleaning
        # remove punctuation, ...
        print('Data cleaning.')

        X_all_text = []
        X_train = text_cleaning(X_train, X_all_text)
        X_valid = text_cleaning(X_valid, [])
        X_test = text_cleaning(X_test, [])


        # prepare label text
        print('Preparing label text')
        for i in tqdm(range(len(train_abstr))):
            train_abstr[i] = text_cleaner(train_abstr[i])
        
        for j in tqdm(range(len(valid_abstr))):
            valid_abstr[j] = text_cleaner(valid_abstr[j])

        print('Add start and end tagger to the labels')
        def add_tagger(abstractive, all_text):
            for i in tqdm(range(len(abstractive))):
                abstractive[i] = '_BOS_ ' + abstractive[i] + ' _EOS_'
                all_text.append(abstractive[i])
            return abstractive

        Y_all_text = []

        train_abstr = add_tagger(train_abstr, Y_all_text)
        valid_abstr = add_tagger(valid_abstr, [])
       

        MAX_LEN = 100
        MAX_SUMMARY_LEN = 30
        MAX_NUM_WORDS = 20000
        EMBEDDING_DIM = 300
        
        # finally, vectorize the text samples into a 2D integer tensor
        print('Tokenization')
        X_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

        # Fit the tokenizer on training text
        X_tokenizer.fit_on_texts(X_all_text)

        # Get all words that the tokenizer knows
        X_word_index = X_tokenizer.word_index
        num_words = min(len(X_word_index) + 1, MAX_NUM_WORDS)
        print('Found %s unique tokens' % len(X_word_index))

        print('text to sequence vector')

        X_train = text_to_sequence(X_train, [], X_tokenizer)
        X_valid = text_to_sequence(X_valid, [], X_tokenizer)
        X_test = text_to_sequence(X_test, [], X_tokenizer)

        X_train = text_to_one_list(X_train)
        X_valid = text_to_one_list(X_valid)
        X_test = text_to_one_list(X_test)

        Y_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        Y_tokenizer.fit_on_texts(train_abstr)
        Y_word_index = Y_tokenizer.word_index

        # Y tokenization only trained on train data label
        Y_train = Y_tokenizer.texts_to_sequences(train_abstr)
        Y_valid = Y_tokenizer.texts_to_sequences(valid_abstr)

        X_train = pad_sequences(X_train, maxlen=MAX_LEN, truncating='post', padding='post') # Each sentence is padding to a size=max_len vector
        X_valid = pad_sequences(X_valid, maxlen=MAX_LEN, truncating='post', padding='post')
        X_test = pad_sequences(X_test, maxlen=MAX_LEN, truncating='post', padding='post')
        Y_train = pad_sequences(Y_train, maxlen=MAX_SUMMARY_LEN, truncating='post', padding='post')
        Y_valid = pad_sequences(Y_valid, maxlen=MAX_SUMMARY_LEN, truncating='post', padding='post')


        # Load pretrained
        # Prepare pre-trained embedding matrix
        np.save('./data/train.npy', X_train)
        np.save('./data/abstractive_label.npy', Y_train)
        with open('./data/X_tokenizer.pkl', 'wb') as file:
            pickle.dump(X_tokenizer, file)
            file.close()
        with open('./data/Y_tokenizer.pkl', 'wb') as file:
            pickle.dump(Y_tokenizer, file)


        return 


if __name__ == "__main__":
    load_data(target='extractive')
    load_data(target='abstractive')






















