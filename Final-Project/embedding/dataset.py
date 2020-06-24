from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import pdb

class Embedding_dataset(Dataset):

    def __init__(self):

        self.data = np.load('./preprocess_data/input_ids.npy', allow_pickle=True)
        self.mask_attention = np.load('./preprocess_data/mask_attention.npy', allow_pickle=True)
        self.label = np.load('./preprocess_data/label.npy', allow_pickle=True)

        self.len = len(self.data)

    def __getitem__(self, idx):

        ids = self.data[idx]
        label = self.label[idx]
        mask_attention = self.mask_attention[idx]

        token_tensor = torch.tensor(ids)
        label_tensor = torch.tensor(label)
        mask_attention_tensor = torch.tensor(mask_attention)

        return token_tensor, mask_attention_tensor, label_tensor

    def __len__(self):
        return self.len


def create_mini_batch(samples):

    token_tensors = torch.stack([s[0] for s in samples])
    mask_attention_tensors = torch.stack([s[1] for s in samples])

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    return token_tensors, mask_attention_tensors, label_ids

