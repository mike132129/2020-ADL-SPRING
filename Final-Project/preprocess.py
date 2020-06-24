
# coding: utf-8

# In[37]:


import pandas as pd
import unicodedata
import re
import os
import pickle

import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

import pdb


# In[2]:


all_tag=[
    '調達年度',  #0
    '都道府県',  #1
    '入札件名',  #2
    '施設名',  #3
    '需要場所(住所)',  #4
    '調達開始日',  #5
    '調達終了日',  #6
    '公告日',  #7
    '仕様書交付期限',  #8
    '質問票締切日時',  #9
    '資格申請締切日時',  #10
    '入札書締切日時',  #11
    '開札日時',  #12
    '質問箇所所属/担当者',  #13
    '質問箇所TEL/FAX',  #14
    '資格申請送付先',  #15
    '資格申請送付先部署/担当者名',  #16
    '入札書送付先',  #17
    '入札書送付先部署/担当者名', #18
    '開札場所'  #19
]


# In[3]:

def get_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    tag_list =[]
    start = 0
    end = 0
    for i, word in enumerate(tag):
        if word != ';' and (i+1)!=len(tag):
            end = end + 1
        elif (i+1) == len(tag):
            tag_list.append(all_tag.index(tag[start:]))
        else:
            tag_list.append(all_tag.index(tag[start:end]))
            end = end + 1
            start = end
    return tag_list


# In[4]:


def get_value(value,text):
    value_list =[]
    start_index = []
    start = 0
    end = 0
    for i, word in enumerate(value):
        if word != ';' and (i+1)!=len(value):
            end = end + 1
        elif (i+1) == len(value):
            value_list.append(value[start:])
        else:
            value_list.append(value[start:end])
            end = end + 1
            start = end
    for i in range(len(value_list)):
        start_index.append(text.find(value_list[i]))
    return value_list, start_index


# In[5]:


path = "train/ca_data"
dirs = os.listdir(path)
data = []
for file in dirs:
    df = pd.read_excel(path + '/' + file)
    for i in range(len(df['Index'])):
        temp_data = {}
    
    
        temp_data['tag'] = []
        if df['Tag'][i] == df['Tag'][i]:
            temp_data['tag'] = get_tag(df['Tag'][i])
       
        temp_data['value'] = []
        temp_data['value_start'] = []
        if df['Value'][i] == df['Value'][i]:   
            value_return = get_value(df['Value'][i], df['Text'][i])
            temp_data['value'] = value_return[0]
            temp_data['value_start'] = value_return[1]
    
        data.append(temp_data)


# In[6]:


from transformers import BertTokenizer
pretrained_bert = "cl-tohoku/bert-base-japanese"
tokenizer = BertTokenizer.from_pretrained(pretrained_bert,do_lower_case=True)


# In[9]:


# tag, value 1 to 1
def get_token_range(tokenizer, sample):
    ranges = []
    assert len(sample['value']) == len(sample['tag'])
    for start in range(len(sample['tag'])):
        char_start = sample['value_start'][start]
        char_end = char_start + len(sample['value'][start])
        token_start = 1 + len(tokenizer.tokenize(sample['text'][:char_start]))
        answer_len = len(tokenizer.tokenize(sample['text'][char_start:char_end]))
        token_end = token_start + answer_len
        ranges.append([token_start,token_end])
    return ranges


# multi tag, one value
def get_token_range_ver2(tokenizer, sample):
    ranges = []
    for start in range(len(sample['tag'])):
        char_start = sample['value_start'][0]
        char_end = char_start + len(sample['value'][0])
        token_start = 1 + len(tokenizer.tokenize(sample['text'][:char_start]))
        answer_len = len(tokenizer.tokenize(sample['text'][char_start:char_end]))
        token_end = token_start + answer_len
        ranges.append([token_start,token_end])
    return ranges

def get_target_and_value(temp_tag, temp_range, tag_id):
    assert len(temp_tag) == len(temp_range)
    tag_exist = 0
    value_range = [0,0]
    if tag_id in temp_tag:
        tag_exist = 1
        value_range = temp_range[temp_tag.index(tag_id)]
        
        
    return tag_exist, value_range


# In[12]:


def process_bert_samples(tokenizer, samples):

    processeds = []
    for sample in tqdm(samples):
        orginal_text = tokenizer.tokenize(sample['text'])
        half_text = (unicodedata.normalize('NFKC', sample['text']))
        half_text = tokenizer.tokenize(half_text)

        text = (tokenizer.encode(half_text)[:-1])
        processed = {}
        processed['text'] = text
        temp_range = []
        temp_tag = sample['tag']
        if len(sample['tag'])!=len(sample['value']):
            if len(sample['tag']) == 1:
                continue
            else:
                temp_range = get_token_range_ver2(tokenizer, sample)
        else:
            temp_range = get_token_range(tokenizer, sample)

        temp_each_tag = {}
        temp_each_tag['tag_0'], temp_each_tag['value_0'] = get_target_and_value(temp_tag, temp_range, 0)
        temp_each_tag['tag_1'], temp_each_tag['value_1'] = get_target_and_value(temp_tag, temp_range, 1)
        temp_each_tag['tag_2'], temp_each_tag['value_2'] = get_target_and_value(temp_tag, temp_range, 2)
        temp_each_tag['tag_3'], temp_each_tag['value_3'] = get_target_and_value(temp_tag, temp_range, 3)
        temp_each_tag['tag_4'], temp_each_tag['value_4'] = get_target_and_value(temp_tag, temp_range, 4)
        temp_each_tag['tag_5'], temp_each_tag['value_5'] = get_target_and_value(temp_tag, temp_range, 5)
        temp_each_tag['tag_6'], temp_each_tag['value_6'] = get_target_and_value(temp_tag, temp_range, 6)
        temp_each_tag['tag_7'], temp_each_tag['value_7'] = get_target_and_value(temp_tag, temp_range, 7)
        temp_each_tag['tag_8'], temp_each_tag['value_8'] = get_target_and_value(temp_tag, temp_range, 8)
        temp_each_tag['tag_9'], temp_each_tag['value_9'] = get_target_and_value(temp_tag, temp_range, 9)
        temp_each_tag['tag_10'], temp_each_tag['value_10'] = get_target_and_value(temp_tag, temp_range, 10)
        temp_each_tag['tag_11'], temp_each_tag['value_11'] = get_target_and_value(temp_tag, temp_range, 11)
        temp_each_tag['tag_12'], temp_each_tag['value_12'] = get_target_and_value(temp_tag, temp_range, 12)
        temp_each_tag['tag_13'], temp_each_tag['value_13'] = get_target_and_value(temp_tag, temp_range, 13)
        temp_each_tag['tag_14'], temp_each_tag['value_14'] = get_target_and_value(temp_tag, temp_range, 14)
        temp_each_tag['tag_15'], temp_each_tag['value_15'] = get_target_and_value(temp_tag, temp_range, 15)
        temp_each_tag['tag_16'], temp_each_tag['value_16'] = get_target_and_value(temp_tag, temp_range, 16)
        temp_each_tag['tag_17'], temp_each_tag['value_17'] = get_target_and_value(temp_tag, temp_range, 17)
        temp_each_tag['tag_18'], temp_each_tag['value_18'] = get_target_and_value(temp_tag, temp_range, 18)
        temp_each_tag['tag_19'], temp_each_tag['value_19'] = get_target_and_value(temp_tag, temp_range, 19)
        processed['target'] = temp_each_tag

            
        processeds.append(processed)
    return processeds


# In[13]:


training_data = process_bert_samples(tokenizer,data)

pdb.set_trace()

# In[18]:


def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(seq[:to_len] + [padding]*max(0, to_len - len(seq)))
    return paddeds


# In[27]:


class BertDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'inputs' : sample['text'],
            'target': sample['target'],
            'segment': [0]*len(sample['text']),
            'mask' : [1] * len(sample['text'])
        }
        return instance
    
    def collate_fn(self, samples):
        batch = {}
        for key in ['target']:
            if any (key not in sample for sample in samples):
                continue
            batch[key] = [sample[key] for sample in samples]
            
        for key in ['inputs', 'segment', 'mask']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len([sample[key] for sample in samples], to_len, 0)
            batch[key] = torch.tensor(padded)
        return batch


# In[39]:


def create_Bert_dataset(samples):
    dataset = BertDataset(samples)
    with open('training.pkl', 'wb') as f:
        pickle.dump(dataset,f)
    return dataset


# In[40]:


train_dataset = create_Bert_dataset(process_bert_samples(tokenizer, data))


# In[41]:


file = open('training.pkl','rb')    
train_dataset = pickle.load(file)      
file.close()


# In[42]:


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 8,
    shuffle = True,
    collate_fn = lambda x: BertDataset.collate_fn(train_dataset, x)
)

