import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import pdb
import tqdm
from argparse import ArgumentParser
from dataset import ques_ans_dataset, create_mini_batch
from makedata import *
from module import modified_bert

parser = ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--test_data_path')
parser.add_argument('--load_model')
parser.add_argument('--output_path')
parser.add_argument('--plot')
parser.add_argument('--threshold')

args = parser.parse_args()


def preprocess_test_data(test_data_path, tokenizer):

    # ONLY used when predicting 
    raw_data = load_data(test_data_path)
    context, question, question_id, _, _ = parse_data(raw_data, mode=args.mode)

    # Concatenate text and question
    # text, text_question_segment = tokenization(context, question, tokenizer)
    text, text_question_segment, text_attention_mask, text_length = tokenizationn(context, question, tokenizer)
    return text, text_question_segment, question_id, text_attention_mask, text_length


def predictt(data_set, BATCH_SIZE, text, question_id, compute_acc=False, threshold=0.5):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)

    # Load model
    model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

    model.to(device)

    cus_model = modified_bert(model)
    cus_model.load_state_dict(torch.load(args.load_model))
    cus_model.to(device)

    dataloader = DataLoader(data_set, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    cls_result = []
    sta_result = []
    end_result = []

    cus_model.eval()
 
    with torch.no_grad():

        for data in tqdm(dataloader):

            if next(cus_model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]

            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors, text_length = data
            outputs = cus_model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,
                            labels=None
                            )

            

            start_logits = outputs[0]
            end_logits = outputs[1]
            cls_logits = outputs[2]

            sigmoid = nn.Sigmoid()

            if not args.threshold:
                threshold = 0.5
            else:
                threshold = float(args.threshold)

            cls_pred = sigmoid(cls_logits)
            if cls_pred > threshold:
                cls_pred = [1]
            else:
                cls_pred = [0]

            softmax = nn.Softmax(dim=1)

            start_logits = start_logits[:, :text_length.item()+1]
            start_pred = softmax(start_logits).argmax(dim=-1)

            end_logits = end_logits[:, :text_length.item()+1]
            end_pred = softmax(end_logits).argmax(dim=-1)

            cls_result += cls_pred
            sta_result += start_pred
            end_result += end_pred

    ans_dict = {}


    with open(args.output_path, 'w') as file:
        for i in range(len(cls_result)):

            if cls_result[i] == 0:
                ans_dict[question_id[i]] = ''

            else:

                start = sta_result[i]
                end = end_result[i]

                if start > end:
                    ans_dict[question_id[i]] = ''


                if end - start > 30:
                    answer = tokenizer.convert_ids_to_tokens(text[i][start:start+30], skip_special_tokens=True)
                    answer = tokenizer.convert_tokens_to_string(answer)
                    answer = answer.replace(' ', '')
                    ans_dict[question_id[i]] = ''


                else:
                    answer = tokenizer.convert_ids_to_tokens(text[i][start:end+1], skip_special_tokens=True)
                    answer = tokenizer.convert_tokens_to_string(answer)
                    answer = answer.replace(' ', '')
                    ans_dict[question_id[i]] = answer

                
        file.write(json.dumps(ans_dict))
        file.close()

    del model
    del cus_model
    return


def main():

    model_version = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)

    text, text_question_segment, question_id, text_attention_mask, text_length = preprocess_test_data(args.test_data_path, tokenizer)

    # Flatten
    question_id = [item for sublist in question_id for item in sublist]

    np.save('./data/test.npy', text)
    np.save('./data/test_segment.npy', text_question_segment)
    np.save('./data/test_text_length.npy', text_length)

    trainset = ques_ans_dataset(mode=args.mode)
    predictt(trainset, BATCH_SIZE=1, text=text, question_id=question_id, compute_acc=False)

if __name__ == '__main__':


    main()



