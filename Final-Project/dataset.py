from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb
import pickle
from utils import normalize_text
from keras.preprocessing.sequence import pad_sequences

class EE_dataset(Dataset):

    def __init__(self, mode):

        self.mode = mode
        assert self.mode in ['train', 'valid', 'test']

        if self.mode ==  'train':   
            self.data = np.load('./train/input_ids.npy', allow_pickle=True)
            self.label = np.load('./train/label.npy', allow_pickle=True)
            self.token_type = np.load('./train/token_type.npy', allow_pickle=True)
            self.mask_attention = np.load('./train/mask_attention.npy', allow_pickle=True)

        elif self.mode == 'valid':
            self.data = np.load('./dev/input_ids.npy', allow_pickle=True)
            self.label = np.load('./dev/label.npy', allow_pickle=True)
            self.token_type = np.load('./dev/token_type.npy', allow_pickle=True)
            self.mask_attention = np.load('./dev/mask_attention.npy', allow_pickle=True)

        else:
            self.data = np.load('./preprocess-test-data/input_ids.npy', allow_pickle=True)
            self.token_type = np.load('./preprocess-test-data/token_type.npy', allow_pickle=True)
            self.mask_attention = np.load('./preprocess-test-data/mask_attention.npy', allow_pickle=True)

        self.len = len(self.data)

    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode =='valid':

            ids = self.data[idx]
            label = self.label[idx]
            token_type = self.token_type[idx]
            mask_attention = self.mask_attention[idx]
            
            tokens_tensor = torch.tensor(ids)
            token_type_tensor = torch.tensor(token_type)
            label_tensor = torch.tensor(label)
            mask_attention_tensor = torch.tensor(mask_attention)

            return tokens_tensor, token_type_tensor, mask_attention_tensor, label_tensor

        else:
            ids = self.data[idx]
            token_type = self.token_type[idx]
            mask_attention = self.mask_attention[idx]
            
            label_tensor = None
            tokens_tensor = torch.tensor(ids)
            token_type_tensor = torch.tensor(token_type)
            mask_attention_tensor = torch.tensor(mask_attention)

            return tokens_tensor, token_type_tensor, mask_attention_tensor, label_tensor   

    def __len__(self):
        return self.len    


def create_mini_batch(samples):

    token_tensors = torch.stack([s[0] for s in samples])
    token_type_tensors = torch.stack([s[1] for s in samples])
    mask_attention_tensors = torch.stack([s[2] for s in samples])

    if samples[0][3] is not None:
        label_ids = torch.stack([s[3] for s in samples])
    else:
        label_ids = None

    return token_tensors, token_type_tensors, mask_attention_tensors, label_ids

'''
https://github.com/HandsomeCao/Bert-BiLSTM-CRF-pytorch/blob/master/utils.py
'''

class NerDataset(Dataset):
    def __init__(self, f_path, tokenizer, tag2idx):
        with open(f_path, 'rb') as file:
            data = pickle.load(file)

        sentences, tags = [], []

        for d in data:
            entries = d.split('\n')

            sentence = [line.split()[0] for line in entries]
            tag = [line.split()[1] for line in entries]

            sentence = ['[CLS]'] + sentence + ['[SEP]']
            tag = ['[CLS]'] + tag + ['[SEP]']

            sentences.append(sentence)
            tags.append(tag)

        
        # for i in range(len(tags)):

        #     for j in reversed(range(2, len(tags[i])-2)):

        #         if len(tags[i][j]) != len(tags[i][j+1]):

        #             tags[i] = tags[i][:j+1] + ['[SEP]'] + ['[CLS]'] + tags[i][j+1:]
        #             sentences[i] = sentences[i][:j+1] + ['[SEP]'] + ['[CLS]'] + sentences[i][j+1:]
        #             continue

        #         else:

        #             if len(tags[i][j]) != 1 and len(tags[i][j]) != 1:

        #                 if tags[i][j].split('-')[1] != tags[i][j+1].split('-')[1]:
        #                     tags[i] = tags[i][:j+1] + ['[SEP]'] + ['[CLS]'] + tags[i][j+1:]
        #                     sentences[i] = sentences[i][:j+1] + ['[SEP]'] + ['[CLS]'] + sentences[i][j+1:]
        #                     continue

        #     assert (len(sentences[i]) == len(tags[i]))

            
            



        # sents, tags_li = [], [] # list of lists
        # for entry in entries:
        #     words = [line.split()[0] for line in entry.splitlines()]
        #     tags = ([line.split()[-1] for line in entry.splitlines()])
        #     if len(words) > MAX_LEN:
        #         # 先对句号分段
        #         word, tag = [], []
        #         for char, t in zip(words, tags):
                    
        #             if char != '。':
        #                 if char != '\ue236':   # 测试集中有这个字符
        #                     word.append(char)
        #                     tag.append(t)
        #             else:
        #                 sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
        #                 tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
        #                 word, tag = [], []            
        #         # 最后的末尾
        #         if len(word):
        #             sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
        #             tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
        #             word, tag = [], []
        #     else:
        #         sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
        #         tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sentences, tags
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
                

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            w = normalize_text(w)
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            is_head = [1] + [0]*(len(xx) - 1)
            
            # t = [t] + ['<PAD>'] * (len(tokens) - 1)

            yy = [self.tag2idx[t]]  # (T,)


            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        
        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    x = f(1)
    y = f(-2)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    x = pad_sequences(x, maxlen=512, dtype="long", truncating="post", padding="post")
    y = pad_sequences(y, maxlen=512, dtype="long", truncating="post", padding="post")

    f = torch.tensor
    return words, f(x), is_heads, tags, f(y), seqlens