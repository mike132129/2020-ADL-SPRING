import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from makedata import *
from tensorflow.python.keras.models import Model, Input, load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pdb
import seaborn
from argparse import ArgumentParser
with open('./data/X_tokenizer.pkl', 'rb') as file:
    X_tokenizer = pickle.load(file)
    file.close()
    
with open('./data/Y_tokenizer.pkl', 'rb') as file:
    Y_tokenizer = pickle.load(file)
    file.close()
X_embedding = np.load('./data/text_embedding.npy')
MAX_LEN = 100
num_words = 20000
EMBEDDING_DIM = 300
MAX_SUMMARY_LEN = 30

parser = ArgumentParser()
parser.add_argument('--plot')
args = parser.parse_args()

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

# Q4: Plot the distribution of relative locations
def relative_locations():
    _, valid_text, valid_sentence_length = process_test_data(test_data_path='./data/valid.jsonl', tokenizer=X_tokenizer)
    _, test_text, test_sentence_length = process_test_data(test_data_path='./data/test.jsonl', tokenizer=X_tokenizer)

    from module import biLSTM_baseline
    model = biLSTM_baseline(embedding_matrix=X_embedding, MAX_LEN=MAX_LEN, 
                            num_words=num_words, EMBEDDING_DIM=EMBEDDING_DIM,
                            LSTM_units=256, LSTM_dropout=0.5)
    print('LOADING MODEL')
    model.load_weights('./model/extractive.hdf5')

    
    def predic(X, sentence_length, density):
        print('Start predicting..')
        pred = model.predict(X)
        pred = np.squeeze(pred)
        print('Post processing')

        for i in tqdm(range(len(X))):
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

            num_sentence = len(sentence_length[i])
            for j in win:
                if j == 0:
                    continue
                density += [int(j)/num_sentence]
        return density

    density = []
    density = predic(valid_text, valid_sentence_length, density)
    density = predic(test_text, test_sentence_length, density)

    fig, ax = plt.subplots() 
    num_bins = 30
    plt.hist(density, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('relative locations')
    plt.ylabel('density')
    plt.show()
    plt.savefig('relative-locations.png')




# Q5: Visualize attention weight
def attention_weight():
    
    reverse_source_word_index = X_tokenizer.index_word
    reverse_target_word_index = Y_tokenizer.index_word
    target_word_index = Y_tokenizer.word_index

    _, valid_text, _ = process_test_data(test_data_path='./data/valid.jsonl', tokenizer=X_tokenizer)
    model = load_model('./model/abstractive_with_att.hdf5')
    attn_layer = model.layers[9]
    attention_model = Model(inputs=model.inputs, outputs=model.outputs + [attn_layer.output])

    encoder_input = valid_text[77]
    decoder_input = np.zeros((len(encoder_input), 30))
    decoder_input[:, 0] = target_word_index['bos']

    for i in range(1, 30):
        output, attention = attention_model.predict([encoder_input.reshape(1, -1), decoder_input])
        decoder_input[0][i] = output.argmax(axis=-1)[0][i-1]
        attention_density = attention[0]
    x = [i for i in encoder_input if i != 0]
    y = [i for i in decoder_input[0][1:] if i != 0]

    x_label = [reverse_source_word_index[i] for i in x]
    y_label = [reverse_target_word_index[int(i)] for i in y]

    sentence_text = ''
    for i in x_label:
        sentence_text += i + ' '

    sentence_sum = ''
    for i in y_label:
        sentence_sum += i + ' '

    print(sentence_text)
    print(sentence_sum)


    print('Plotting')
    
    plt.clf()
    plt.figure(figsize=(95, 40))
    seaborn.set(font_scale=6)
    ax = seaborn.heatmap(attention_density[:len(y_label), :len(x_label) + 1], 
            xticklabels=[w for w in x_label],
            yticklabels=[w for w in y_label],
            cbar=False)
    plt.show()
    plt.savefig('attn.png')
    print('Save')
    return

if args.plot == 'relative_locations':
    relative_locations()
else:
    attention_weight()




