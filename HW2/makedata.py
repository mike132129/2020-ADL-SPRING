import torch
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import pdb
from argparse import ArgumentParser

# Load data 
import json


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if np.array_equal(l[ind:ind+sll], sl): # check array is equal
            results.append((ind,ind+sll-1))

    return results



def load_data(data_path):
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    return raw_data


def parse_data(data, mode):
    print('Parse data..')
    context = []
    qas = []
    qas_id = []
    answer = []
    answerable = []

    assert mode in ['train', 'test']

    if mode == 'train':
        for i in range(len(data['data'])): # title index
            for j in range(len(data['data'][i]['paragraphs'])):
                context += [data['data'][i]['paragraphs'][j]['context']]
                
                question_of_this_context = []
                question_id_this_context = []
                answer_of_this_context = []
                answerable_of_this_context = []
                for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                    
                    question_id_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['id']]
                    question_of_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['question']]
                    answer_of_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']]
                                               
                    answerable_of_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['answerable']]
                qas += [question_of_this_context]
                qas_id += [question_id_this_context]
                answer += [answer_of_this_context]
                answerable += [answerable_of_this_context]

    else:
        for i in range(len(data['data'])): # title index
            for j in range(len(data['data'][i]['paragraphs'])):
                context += [data['data'][i]['paragraphs'][j]['context']]
                
                question_of_this_context = []
                question_id_this_context = []
               
                for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                    
                    question_id_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['id']]
                    question_of_this_context += [data['data'][i]['paragraphs'][j]['qas'][k]['question']]
                    
                qas += [question_of_this_context]
                qas_id += [question_id_this_context]
                


    # if the mode is test, return empty answer and answerable                                       
    return context, qas, qas_id, answer, answerable


def tokenization(context, question, tokenizer):

    print('Tokenization')

    CONTEXT_MAX_LEN = 474
    QUESTION_MAX_LEN = 35

    text = []
    text_question_segment = []
    print(len(question))
    for i in tqdm(range(len(question))):

        token_text = tokenizer.tokenize(context[i])
        if len(token_text) > CONTEXT_MAX_LEN:
            token_text = token_text[:CONTEXT_MAX_LEN]
        else:
            token_text += (CONTEXT_MAX_LEN - len(token_text)) * ['[PAD]']

        for j in range(len(question[i])):

            token_question = tokenizer.tokenize(question[i][j])

            if len(token_question) > QUESTION_MAX_LEN:
                token_question = token_question[:QUESTION_MAX_LEN]

            else:
                token_question += (QUESTION_MAX_LEN - len(token_question)) * ['[PAD]']

            word_pieces = ['[CLS]']
            word_pieces += token_text + ['[SEP]']
            len_context = len(word_pieces)
            
            
            word_pieces += token_question + ['[SEP]']
            len_question = len(word_pieces) - len_context
            
            ids = np.asarray(tokenizer.convert_tokens_to_ids(word_pieces))
            segment = np.asarray([0] * len_context + [1] * len_question)

            assert len(word_pieces) == 512
            
            text += [ids]
            text_question_segment += [segment]

    return text, text_question_segment



def tokenizationn(context, question, tokenizer):

    print('Tokenization')

    text = []
    text_length = []
    text_question_segment = []
    text_attention_mask = []
    label = []

    for i in tqdm(range(len(question))):

        _answerable = []

        token_text = tokenizer.tokenize(context[i])
        token_text_id = tokenizer.convert_tokens_to_ids(token_text)
 
        for j in range(len(question[i])):

            token_question = tokenizer.tokenize(question[i][j])
            token_question_id = tokenizer.convert_tokens_to_ids(token_question)

            # combine is a dictionary containing input ids, token type, attention mask
            combine = tokenizer.prepare_for_model(ids=token_text_id, pair_ids=token_question_id, 
                                                max_length=512, truncation_strategy='only_first', 
                                                pad_to_max_length=True, return_overflowing_tokens=True
                                                )
            


            input_ids = np.array(combine['input_ids'])
            token_types_ids = np.array(combine['token_type_ids'])
            attention_mask = np.array(combine['attention_mask'])

            text += [input_ids]
            try:
                truncate_length = combine['num_truncated_tokens']
                text_length += [len(token_text_id) - truncate_length]
            except:
                text_length += [len(token_text_id)]

            text_question_segment += [token_types_ids]
            text_attention_mask += [attention_mask]


    return text, text_question_segment, text_attention_mask, text_length


def make_label(answerable, answer, context):

    print('Make label')
    
    label = []

    iterator = iter(context)

    for i in range(len(answerable)):

        _answerable = []
        
        for j in range(len(answerable[i])):

            if not answer[i][j]: # answer is empty string
                start, end = None, None
                next(iterator)

            else:
                text = next(iterator)

                token_answer = tokenizer.tokenize(answer[i][j])
                token_answer = tokenizer.convert_tokens_to_ids(token_answer)
                ans_range = find_sub_list(token_answer, text)
                
                if not ans_range:
                    start, end = None, None

                else:
                    start, end = ans_range[0]

            if answerable[i][j] == True:
                _answerable.append([1, start, end])
            else:
                _answerable.append([0, start, end])

        label += _answerable

    label = np.array(label, dtype=np.float)

    return label


def save_data_preprocessing(context, context_question_segment, context_attention_mask, text_length, label):

    # ONLY used for training
    np.save('./data/' + args.dataset + '.npy', context)
    np.save('./data/' + args.dataset + '_segment.npy', context_question_segment)
    np.save('./data/' + args.dataset + '_attention_maske.npy', context_attention_mask)
    np.save('./data/' + args.dataset + '_text_length.npy', text_length)
    np.save('./data/' + args.dataset + '_label.npy', label)

    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()

    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    print('Data Preprocess for Training Data')
    raw_data = load_data('./data/' + args.dataset + '.json')
    context, question, question_id, answer, answerable = parse_data(raw_data, mode='train')
    text, text_question_segment, text_attention_mask, text_length = tokenizationn(context, question, tokenizer)
    # text, text_question_segment = tokenization(context, question, tokenizer)
    label = make_label(answerable, answer, text)
    save_data_preprocessing(text, text_question_segment, text_attention_mask, text_length, label)




























