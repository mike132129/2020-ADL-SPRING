import csv
import unicodedata
import re
import pandas as pd
import pdb
import glob, os
from ohiyo import *
import argparse
from tqdm import tqdm
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


'''
send a number to tell what bert is reading the sentence

multilabel -> nn.MultiLabelSoftMarginLoss

'''

def parse():
    parser = argparse.ArgumentParser(description="make data")
    parser.add_argument('--data_dir', default='train/ca_data', type=str,
                        help='data_path_to_pdf')
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)
    args = parser.parse_args()
    return args

def normalize_text(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if np.array_equal(l[ind:ind+sll], sl): # check array is equal
            results.append((ind,ind+sll-1))

    return results

def data_to_paragraph(args):
    '''
    load data return a list containing many dictionary.
    each dictionary is a paragraph from pdf
    assure that
    1. that the length is not more than 500
    2. each tag and value is included in the paragraph

    '''
    path = args.data_dir
    dirs = os.listdir(path)
    paragraph_data = []

    data = []

    for file in tqdm(dirs):
        tmp_data = {}
        tmp_data['text'] = ''
        tmp_data['tag'] = []
        try:
            df = pd.read_excel(path + '/' + file)
        except Exception as e:
            print("error found: ", e)
        paragraph = 1

        for line in range(len(df)):

            text = normalize_text(df.iloc[line]['Text'])
            if len(text) + len(tmp_data['text']) > 500:
                tmp_data['paragraph'] = paragraph
                paragraph += 1
                data.append(tmp_data.copy())
                paragraph_data.append(tmp_data['text'])
                tmp_data.clear()

                # setting new dictionary
                tmp_data['text'] = ''
                tmp_data['tag'] = []

            tmp_data['text'] += text

            if pd.notna(df.iloc[line]['Tag']): #  if there are tag(s) in this line
                tags = normalize_text(df.iloc[line]['Tag']).split(';')
                values = normalize_text(df.iloc[line]['Value']).split(';')
                for tag, value in zip(tags, values):
                    tmp_data['tag'].append([tag, value])

        tmp_data['paragraph'] = paragraph
        data.append(tmp_data.copy())
        tmp_data.clear()

    # import pickle 
    # with open('data.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    # pdb.set_trace()

    return data

def tag_to_paragraph(data):
    '''
    This function convert raw data into tag-to-text format
    for the convenience for bert model(?
    '''
    each_tag = []

    for paragraph in tqdm(data):

        tmp_tag = {}
        if not paragraph['tag']:
            tmp_tag['tag'] = ''
            tmp_tag['text'] = paragraph['text']
            tmp_tag['paragraph'] = paragraph['paragraph']
            each_tag.append(tmp_tag.copy())
            tmp_tag.clear()

        for tag in paragraph['tag']:
            tmp_tag['tag'] = tag[0]
            tmp_tag['value'] = tag[1]
            tmp_tag['text'] = paragraph['text']
            tmp_tag['paragraph'] = paragraph['paragraph']
            each_tag.append(tmp_tag.copy())
            tmp_tag.clear()

    return each_tag

def create_token_type(input_id):
    state = 'tag' # state is tag, paragraph, text or padding
    token_type_id = []
    for idx in input_id:
        
        if idx == 3 and state == 'tag':
            token_type_id += [0]
            state = 'paragraph'
            continue

        if state == 'tag':
            token_type_id += [0]

        if idx == 3 and state == 'paragraph':
            token_type_id += [1]
            state = 'text'
            continue

        if state == 'paragraph':
            token_type_id += [1]

        if idx == 3 and state == 'text':
            token_type_id += [1]
            state = 'padding'
            continue

        if state == 'text':
            token_type_id += [1]

        if state == 'padding':
            token_type_id += [0]

    assert len(input_id) == len(token_type_id)

    return token_type_id

def create_mask(input_id):

    return [1 if idx != 0 else 0 for idx in input_id]

def tokenization(tokenizer, tag_paragraph):

    text = []
    token_type = []
    mask_attention = []


    for data in tag_paragraph:


        parag_text = tokenizer.encode(text=str(data['paragraph']), text_pair=data['text'])[1:]
        tag_parag_text = tokenizer.encode(text=data['tag'], text_pair=parag_text)[:-1]
        text += [tag_parag_text]

    input_ids = pad_sequences(text, maxlen=512, dtype="long", truncating="post", padding="post")
    for input_id in input_ids:
        token_type += [create_token_type(input_id)]
        mask_attention += [create_mask(input_id)]

    return input_ids, np.array(token_type), np.array(mask_attention)
    
def make_label(input_ids, tag_paragraph, tokenizer):

    label = []
    for input_id, tag_value in zip(input_ids, tag_paragraph):
        if not tag_value['tag']:
            label.append([None, None])
            continue
        value = tag_value['value']
        value_id = tokenizer.encode(value, add_special_tokens=False)
        
        find = False
        start = 0
        span = []

        while not find:

            try:
                span = list(find_sub_list(value_id[start:], input_id)[0])
                span[0] -= start
                find = True

            except:
                start += 1
            '''
            TODO:
            Some value cannot find by tracing the input_id
            

            '''
        if span:
            label.append([span[0], span[1]])
        else:
            label.append([None, None])
    label = np.array(label, dtype=np.float)

    return label


if __name__ == '__main__':
    args = parse()

    if args.bert:
        model_version = 'cl-tohoku/bert-base-japanese'
        tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    raw_data = data_to_paragraph(args)
    tag_paragraph = tag_to_paragraph(raw_data)
    input_ids, token_type, mask_attention = tokenization(tokenizer, tag_paragraph)
    label = make_label(input_ids, tag_paragraph, tokenizer)
    if args.train:
        np.save('./train/input_ids.npy', input_ids)
        np.save('./train/token_type.npy', token_type)
        np.save('./train/mask_attention.npy', mask_attention)
        np.save('./train/label.npy', label)
    elif args.valid:
        np.save('./dev/input_ids.npy', input_ids)
        np.save('./dev/token_type.npy', token_type)
        np.save('./dev/mask_attention.npy', mask_attention)
        np.save('./dev/label.npy', label)
    