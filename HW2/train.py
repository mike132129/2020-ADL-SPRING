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
import tqdm
from argparse import ArgumentParser
from dataset import ques_ans_dataset, create_mini_batch
from makedata import *
from utils import format_time
import time
from module import modified_bert
import random 

torch.manual_seed(1)

parser = ArgumentParser()
parser.add_argument('--mode')
args = parser.parse_args()

def preprocess_test_data(test_data_path, tokenizer):

    # ONLY used when predicting 
    raw_data = load_data(test_data_path)
    context, question, question_id, _, _ = parse_data(raw_data, mode=args.mode)

    # Concatenate text and question
    text, text_question_segment = tokenization(context, question, tokenizer)

    return text, text_question_segment, question_id

def train(trainset, validset, BATCH_SIZE):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    # tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # Load model
    model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    model.to(device)
    cus_model = modified_bert(model)
    cus_model.to(device)

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
    valiloader = DataLoader(validset, batch_size=128, collate_fn=create_mini_batch)
    
    epochs = 20
    total_steps = len(dataloader) * epochs

    optimizer = AdamW(cus_model.parameters(), lr=5e-7, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    

    for epoch in range(epochs):

        t0 = time.time()

        print('training new epoch!')

        total_cls_loss = 0.0
        total_sta_loss = 0.0
        total_end_loss = 0.0

        total_loss = [total_sta_loss, total_end_loss, total_cls_loss]
        model.train()
        cus_model.train()

        ####### FREEZE LAYER
        # print('FREEZEEEEEEE')
        for i, [j, k] in enumerate(cus_model.named_parameters()):
            if i < 21:
                print(i, j)
                k.requires_grad = False
        ###########


        for step, data in tqdm(enumerate(dataloader)):

            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))
                print('  Total Start Loss: {}, End Loss: {}, CLS Loss: {}, total: {}'.format(total_loss[0]/step, total_loss[1]/step, total_loss[2]/step, sum(total_loss)/step))


            tokens_tensors, segments_tensors, masks_tensors, labels, _ = [t.to(device) for t in data if t is not None]

            # Forward Propagation
            loss, cls_logits, start_logits, end_logits = cus_model(
                                                                    input_ids=tokens_tensors, 
                                                                    token_type_ids=segments_tensors, 
                                                                    attention_mask=masks_tensors,
                                                                    labels=labels
                                                                    )

            for i in range(len(loss)):
                total_loss[i] += loss[i].item()

            loss = (loss[0] + loss[1] + loss[2])

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

        val_cls_loss = 0.0
        val_sta_loss = 0.0
        val_end_loss = 0.0

        val_loss = [val_sta_loss, val_end_loss, val_cls_loss]
        model.eval()
        cus_model.eval()

        with torch.no_grad():

            for data in tqdm(valiloader):

                tokens_tensors, segments_tensors, masks_tensors, labels, _ = [t.to(device) for t in data if t is not None]
                loss, cls_logits, start_logits, end_logits = cus_model(
                                                                        input_ids=tokens_tensors, 
                                                                        token_type_ids=segments_tensors, 
                                                                        attention_mask=masks_tensors,
                                                                        labels=labels
                                                                        )
                for i in range(len(loss)):
                    val_loss[i] += loss[i].item()




        torch.save(cus_model.state_dict(), './model/non-freeze-epoch-%s.pth' % epoch)
        print('  Total Start Loss: {}, End Loss: {}, CLS Loss: {}'.format(val_loss[0]/len(valiloader), val_loss[1]/len(valiloader), val_loss[2]/len(valiloader)))


def predict(trainset, BATCH_SIZE, question_id, compute_acc=False):

    NUM_LABELS = 1
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model.to(device)
    model.load_state_dict(torch.load(args.load_model))



    predictions = None
    correct = 0
    total = 0

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    print('start predicting')

    result = []

 
    with torch.no_grad():
        # 遍巡整個資料集
        for data in tqdm(dataloader):
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)

            logits = outputs[0]
            
            pred = logits.round().view(-1)

            result += pred
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    ans_dict = {}

    with open(args.output_path, 'w') as file:
        for i in range(len(result)):

            if result[i] == 1:
                ans_dict[question_id[i]] = '有答案'

            else :
                ans_dict[question_id[i]] = ''

        file.write(json.dumps(ans_dict))
        file.close()


    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def main():
    
    trainset = ques_ans_dataset(mode=args.mode)
    validset = ques_ans_dataset(mode='valid')
    train(trainset, validset, BATCH_SIZE=4)

main()




