from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb

class ques_ans_dataset(Dataset):
    def __init__(self, mode):

        self.mode = mode

        assert self.mode in ['train', 'valid', 'test']

        if self.mode ==  'train':   
            self.data = np.load('./data/train.npy', allow_pickle=True)
            self.label = np.load('./data/train_label.npy', allow_pickle=True)
            self.segment = np.load('./data/train_segment.npy', allow_pickle=True)
            self.text_length = np.load('./data/train_text_length.npy', allow_pickle=True)

        elif self.mode == 'valid':
            self.data = np.load('./data/dev.npy', allow_pickle=True)
            self.label = np.load('./data/dev_label.npy', allow_pickle=True)
            self.segment = np.load('./data/dev_segment.npy', allow_pickle=True)
            self.text_length = np.load('./data/dev_text_length.npy', allow_pickle=True)

        else:
            self.data = np.load('./data/test.npy', allow_pickle=True)
            self.segment = np.load('./data/test_segment.npy', allow_pickle=True)
            self.text_length = np.load('./data/test_text_length.npy', allow_pickle=True)

        self.len = len(self.data)

    
    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode =='valid':

            ids = self.data[idx]
            label = self.label[idx]
            segment = self.segment[idx]
            text_length = self.text_length[idx]
            
            tokens_tensor = torch.tensor(ids)
            segment_tensor = torch.tensor(segment)
            label_tensor = torch.tensor(label)
            text_length = torch.tensor(text_length)

            
            return tokens_tensor, segment_tensor, label_tensor, text_length

        else:
            ids = self.data[idx]
            segment = self.segment[idx]
            text_length = self.text_length[idx]
            
            label_tensor = None
            tokens_tensor = torch.tensor(ids)
            segment_tensor = torch.tensor(segment)
            text_length = torch.tensor(text_length)

            return tokens_tensor, segment_tensor, label_tensor, text_length        
        
    def __len__(self):
        return self.len
    

def create_mini_batch(samples):

    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    text_length = [s[3] for s in samples]



    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None


    tokens_tensors = torch.stack(tokens_tensors)
    segments_tensors = torch.stack(segments_tensors)
    text_length_tensor = torch.stack(text_length)


    # tokens_tensors = pad_len(tokens_tensors, 512)
    # segments_tensors = pad_len(segments_tensors, 512)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids, text_length_tensor







