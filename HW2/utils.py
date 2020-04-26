import datetime
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import pdb
import tqdm
from argparse import ArgumentParser
from dataset import ques_ans_dataset, create_mini_batch
from makedata import *
from module import modified_bert
from predict import predictt, preprocess_test_data
import matplotlib.pyplot as plt

parser = ArgumentParser()

parser.add_argument('--plot')
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
parser.add_argument('--mode')
parser.add_argument('--load_model')

args = parser.parse_args()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def answer_length_distribution():
    
    # Predict and output answer file
    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
    
    text, text_question_segment, question_id, text_attention_mask, text_length = preprocess_test_data('./data/train.json', tokenizer)
    question_id = [item for sublist in question_id for item in sublist]

    np.save('./data/test.npy', text)
    np.save('./data/test_segment.npy', text_question_segment)
    np.save('./data/test_text_length.npy', text_length)

    trainset = ques_ans_dataset(mode='test')
    predictt(trainset, BATCH_SIZE=1, text=text, question_id=question_id, compute_acc=False)

    with open(args.output_path, 'r') as file:
        ans = json.load(file)

    length = []
    for i in ans.values():
        length += [len(i)]

    length.sort()

    # remove all zero element
    length = list(filter(lambda a: a != 0, length))

    n_bins = 30
    fig, ax = plt.subplots(figsize=(12, 4))
    n, bins, patches = ax.hist(length, n_bins, density=True, histtype='step', rwidth=0.8,
                               cumulative=True)

    ax.grid(True)
    ax.set_xlabel('Length')
    ax.set_ylabel('count (%)')
    plt.savefig('cumulative-answer-length.png')
    plt.show()

    return 

def answerable_threshold():
    f1 = []
    EM = []
    x = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i in x:
        with open('./score-'+str(i)+'.json', 'r') as file:
            score = json.load(file)
            f1 += [[score['overall']['f1'], score['answerable']['f1'], score['unanswerable']['f1']]]
            EM += [[score['overall']['em'], score['answerable']['em'], score['unanswerable']['em']]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Performance on Different Threshold')
    ax1.plot(x, f1[:, 0])
    ax1.plot(x, f1[:, 1])
    ax1.plot(x, f1[:, 2])
    ax1.set_xlabel('answerable threshold')
    ax2.plot(x, EM[:, 0])
    ax2.plot(x, EM[:, 1])
    ax2.plot(x, EM[:, 2])
    ax2.set_xlabel('answerable threshold')

    plt.savefig('performance-on-answerable-threshold.png')
    plt.show()

    return

if __name__ == '__main__':

    

    if args.plot == 'answer_length':
        answer_length_distribution()

    if args.plot == 'answerable_threshold':
        answerable_threshold()


