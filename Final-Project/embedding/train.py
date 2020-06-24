import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertForMaskedLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
import pdb
from argparse import ArgumentParser
from dataset import Embedding_dataset, create_mini_batch
from utils import setting
import random 
from tqdm.auto import tqdm

torch.manual_seed(1320)

def parse():
    parser = ArgumentParser(description="train")

    parser.add_argument('--batch_size', default=2, type=int)
    args = parser.parse_args()
    return args

def train(trainset, device, model, BATCH_SIZE):

    print('len of training set {}'.format(len(trainset)))
    trainset, validset = torch.utils.data.random_split(trainset, [120, 17])

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, shuffle=True)
    valiloader = DataLoader(validset, batch_size=2, collate_fn=create_mini_batch)

    epochs = 10
    total_steps = len(dataloader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


    for epoch in range(epochs):

        running_loss = 0.0

        model.train()

        ####### FREEZE LAYER
        # print('FREEZEEEEEEE')
        # for i, [j, k] in enumerate(cus_model.named_parameters()):
        #     if i < 21:
        #         print(i, j)
        #         k.requires_grad = False
        ###########

        for step, data in tqdm(enumerate(dataloader)):

            if step % 2000 == 0 and not step == 0:
                print('loss: {}'.format(running_loss/step))

            token_tensors, mask_attention, labels = [t.to(device) for t in data if t is not None]

            pdb.set_trace()

            loss, score = model(input_ids=token_tensors.long(), attention_mask=mask_attention.long(), masked_lm_labels=labels.long())
            running_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)

            # Update parameters
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()

            model.zero_grad()


        print('Validation..')

        val_loss = 0.0

        model.eval()

        with torch.no_grad():
            for data in tqdm(valiloader):
                if next(model.parameters()).is_cuda:
                    token_tensors, mask_attention, labels = [t.to(device) for t in data if t is not None]
                loss, score = model(input_ids=token_tensors.long(), attention_mask=mask_attention.long(), masked_lm_labels=labels.long())
                val_loss += loss.item()

        print('Total Loss: {}'.format(val_loss/len(valiloader)))
        model.save_pretrained('adl-pretrained-model/bert-embedding-epoch-%s/' % epoch)  
        
def main():
    args = parse()
    # model_version = 'cl-tohoku/bert-base-japanese'
    model_version = 'model/bert0621-epoch-3/'

    trainset = Embedding_dataset()
    model, device = setting(model_version)
    train(trainset, device, model, BATCH_SIZE=args.batch_size)

if __name__ == '__main__':
    main()