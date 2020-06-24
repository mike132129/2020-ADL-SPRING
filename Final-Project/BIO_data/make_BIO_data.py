import argparse
import os
import re
import json
import pickle
from define import SEARCH_LIST
import pdb


def parse():
    parser = argparse.ArgumentParser(description="ADL json converter")
    parser.add_argument('--output_path', type=str,
                        help='path_to_output')
    parser.add_argument('--pickle_path', type=str,
                        help='path_to_pickle')
    args = parser.parse_args()
    return args

def find_tag_span(text, value):

    start = text.find(value, 0)
    if start == -1:
        print('there might be something wrong...')
        print('text:{}\nvalue:{}'.format(text, value))
        return 0, 0
    end = start + len(value)
    return start, end

def process_tag(start, end, processed, tag):
    tag = str(tag)

    for i in range(end - start):
        if i == 0:
            if processed[start] == 'O':
                processed[start] = 'B-'+ tag
            else:
                continue
                processed[start] = processed[start] + ' ' + 'B-'+ tag
        else:
            if processed[start + i] == 'O':
                processed[start + i] = 'I-' + tag
            else:
                continue
                processed[start + i] = processed[start + i] + ' ' + 'I-' + tag
    return processed
    

def load_pickle(pickle_path):
    '''
    data(list):
        'text': (str)text,
        'tag': (list) [(str)tag_name, (str)tag_value]
        'paragraph': (int)paragraph
    '''
    tag_dict = SEARCH_LIST.tag_dict
    with open(pickle_path, 'rb') as f:
          data = pickle.load(f)

    out_pickle = []
    for i, sample in enumerate(data):
        processed = ['O'] * len(sample['text'])
        for j, tag in enumerate(sample['tag']):
            tag_name = tag[0]
            tag_value = tag[1]

            if tag_name not in tag_dict:
                continue
 
            start, end = find_tag_span(sample['text'], tag_value)
            if start + end == 0:
                print('fail in data: {} paragraph {}'.format(i, sample['paragraph']))
                exit(0)

            processed = process_tag(start, end, processed, tag_dict[tag_name])

        if processed == ['O'] * len(sample['text']):
            continue
        BIO_data = '\n'.join([sample['text'][k] + ' ' + processed[k] for k in range(len(sample['text']))])

        out_pickle.append(BIO_data)
        
        
    
    return out_pickle





def main(args):
    
    out_pickle = load_pickle(args.pickle_path)
    with open(args.output_path, 'wb') as f:
        pickle.dump(out_pickle, f)

if __name__ == '__main__':
    args = parse()
    main(args)