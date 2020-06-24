import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
import pdb
from argparse import ArgumentParser
from dataset import EE_dataset, create_mini_batch, NerDataset, pad
from makedata import *
from utils import setting
import time
from module import modified_bert, Bert_BiLSTM_CRF
import random
from tqdm.auto import tqdm
import torch.optim as optim
from ohiyo import tag2idx, idx2tag
from sklearn.metrics import f1_score

torch.manual_seed(1320)

def parse():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--use_pretrained_embedding', action='store_true', default=False)
    parser.add_argument('--bert_bilstm_crf', action='store_true', default=False)
    args = parser.parse_args()
    return args

def train(trainset, validset, device, model, cus_model, BATCH_SIZE):

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, shuffle=False)
    valiloader = DataLoader(validset, batch_size=30, collate_fn=create_mini_batch)

    epochs = 20
    total_steps = len(dataloader) * epochs
    optimizer = AdamW(cus_model.parameters(), lr=5e-6, eps=1e-8, weight_decay=0.0003)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    for epoch in range(epochs):

        total_start_loss = 0.0
        total_ennnd_loss = 0.0

        total_loss = [total_start_loss, total_ennnd_loss]

        model.train()
        cus_model.train()

        ###### FREEZE LAYER
        print('FREEZEEEEEEE')
        for i, [j, k] in enumerate(cus_model.named_parameters()):
            if i < 21:
                print(i, j)
                k.requires_grad = False
        ##########

        for step, data in tqdm(enumerate(dataloader)):

            if step % 100 == 0 and not step == 0:
                print('start loss: {}, end loss: {}'.format(total_loss[0]/step, total_loss[1]/step))

            token_tensors, token_type_tensors, mask_attention, labels = [t.to(device) for t in data if t is not None]


            loss = cus_model(input_ids=token_tensors.long(),
                             token_type_ids=token_type_tensors.long(),
                             attention_mask=mask_attention.long(),
                             labels=labels
                             )

            total_loss[0] += loss[0]
            total_loss[1] += loss[1]

            loss = loss[0] + loss[1]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cus_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            
            # Update learning rate schedule
            scheduler.step()

            cus_model.zero_grad()
            model.zero_grad()


        print('Validation..')

        val_sta_loss = 0.0
        val_end_loss = 0.0

        val_loss = [val_sta_loss, val_end_loss]
        model.eval()
        cus_model.eval()

        with torch.no_grad():
            for data in tqdm(valiloader):
                token_tensors, token_type_tensors, mask_attention, labels = [t.to(device) for t in data if t is not None]
                loss = cus_model(
                                input_ids=token_tensors.long(), 
                                token_type_ids=token_type_tensors.long(), 
                                attention_mask=mask_attention.long(),
                                labels=labels
                                )
                val_loss[0] += loss[0]
                val_loss[1] += loss[1]

        print('Total Start Loss: {}, End Loss: {}'.format(val_loss[0]/len(valiloader), val_loss[1]/len(valiloader)))        
        torch.save(cus_model.state_dict(), './model/bert-final-pretrained-embedding-freeze-epoch-%s.pth' % epoch)


def train_with_crf(model, trainset, validset, device, BATCH_SIZE):

    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=pad, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    vhjbhadlk = []
    for epoch in range(1, epochs):

        model.train()
        for i, batch in enumerate(trainloader):
    
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            y = y.to(device)
            _y = y
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(x.long(), y.long())

            loss.backward()

            optimizer.step()

            if i%10==0: # monitoring
                print(f"step: {i}, loss: {loss.item()}")


        ###
        # TODO: validation
        model.eval()
        
        with torch.no_grad():

            Y, Y_hat = [], []

            for i, batch in enumerate(validloader):

                words, x, is_heads, tags, y, seqlens = batch
                x = x.to(device)

                _, y_hat = model(x.long())

                Y.extend(y.numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())

            Y, Y_hat = np.array(Y).reshape(-1), np.array(Y_hat).reshape(-1)
            f_score = f1_score(Y, Y_hat, average='weighted')

            print('f1 score: {} in epoch: {}'.format(f_score, epoch))

        torch.save(model.state_dict(), './model/testbert_bilstm_crf-%s.pth' % epoch)


def main():
    args = parse()
    if args.use_pretrained_embedding:
        model_version = 'embedding/adl-pretrained-model/bert-embedding-epoch-9/'
    else:
        model_version = 'cl-tohoku/bert-base-japanese'

   
    model, cus_model, device = setting(model_version, args)

    if args.bert:
        trainset = EE_dataset(mode='train')
        validset = EE_dataset(mode='valid')
        train(trainset, validset, device, model, cus_model, BATCH_SIZE=args.batch_size)

    elif args.bert_bilstm_crf:
        tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', do_lower_case=True)
        trainset = NerDataset('./BIO_data/BIO_data.pkl', tokenizer, tag2idx)
        validset = NerDataset('./BIO_data/BIO_data_dev.pkl', tokenizer, tag2idx)
        train_with_crf(model, trainset, validset, device, 5)

        



if __name__ == '__main__':
    main()