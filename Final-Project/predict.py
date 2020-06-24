import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch.nn as nn
import pdb
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import EE_dataset, create_mini_batch
from makedata import *
from utils import setting, normalize_text
import re, unicodedata
from module import modified_bert, Bert_BiLSTM_CRF
from ohiyo import *
import os, glob
from makedata import create_token_type, create_mask, tokenization
from ohiyo import idx2tag


def parse():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--data_dir', default='dev/ca_data', type=str,
                        help='data_path_to_pdf')
    parser.add_argument('--output_path', default='./dev/dev_predict.csv', type=str,
                        help='data_path_to_output_json')
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--bert_bilstm_crf', action='store_true', default=False)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()
    return args

def data_to_paragraph(args, tokenizer):


    path = args.data_dir
    dirs = os.listdir(path)
    print('loading file from:{}'.format(path))

    data = []
    file_name = []
    clean_data = {}
    clean_data['clean_text'] = []
    clean_data['line'] = []
    clean_data['paragraph'] = []
    clean_data['doc_id'] = []
    clean_data['raw_text'] = []
    clean_data['tokenized_text'] = []


    print('dir:', dirs)

    for file in dirs:

        tmp_data = {}
        tmp_data['text'] = ''

        df = pd.read_excel(path + '/' + file)
        paragraph = 1

        

        for line in range(len(df)):

            raw_text = df.iloc[line]['Text']

            clean_data['raw_text'].append(raw_text)
            text = normalize_text(raw_text)
            clean_data['clean_text'].append(text)

            tokenized_text = tokenizer.tokenize(text)
            for i in range(len(tokenized_text)):
                if len(tokenized_text) >= 3:
                    if tokenized_text[i][0] == '#' and tokenized_text[i][1] == '#':
                        tokenized_text[i] = tokenized_text[i][2:]

            # clean_data['tokenized_text'].append(''.join(tokenized_text))
            # clean_data['line'].append(line+1)
            # clean_data['paragraph'].append(paragraph)
            # clean_data['doc_id'].append(file.split('.')[0])


            if len(text) + len(tmp_data['text']) > 500:
                tmp_data['paragraph'] = paragraph
                data.append(tmp_data.copy())
                tmp_data.clear()
                # setting new dictionary
                tmp_data['text'] = ''
                paragraph += 1


            clean_data['tokenized_text'].append(''.join(tokenized_text))
            clean_data['line'].append(df.iloc[line]['Index'])
            clean_data['paragraph'].append(paragraph)
            clean_data['doc_id'].append(file.split('.')[0])

            tmp_data['text'] += text

        tmp_data['paragraph'] = paragraph

        data.append(tmp_data.copy())
        tmp_data.clear()



    clean_data_df = pd.DataFrame(clean_data)
    clean_data_df.to_csv('./preprocess-test-data/clean_data_frame.csv', index=False)

    return data

def tag_to_paragraph(data):

    each_tag = []

    for paragraph in data:
        tmp_tag = {}
        for tag in all_tag:
            tmp_tag['tag'] = tag
            tmp_tag['text'] = paragraph['text']
            tmp_tag['paragraph'] = paragraph['paragraph']
            each_tag.append(tmp_tag.copy())
            tmp_tag.clear()

        assert(len(each_tag) % 20 == 0)

    return each_tag


def BeInDateTimeFormat(text):
    regex = '.{2}[0-9１２３４５６７８９０]+年[0-9１２３４５６７８９０]+月[0-9１２３４５６７８９０]+日'
    if re.search(regex, text):
        return True
    return False

def BeInTelTaxFormat(text):
    regex = '\w{2}[-－]\w{4}[-－]\w{4}', '\w{3}[-－]\w{4}[-－]\w{4}', '\w{4}[-－]\w{4}[-－]\w{4}', '\w{3}[-－]\w{3}[-－]\w{4}', '\w{4}[-－]\w{2}[-－]\w{4}'
    if re.search(regex[0], text) or re.search(regex[1], text) or re.search(regex[2], text) or re.search(regex[3], text) or re.search(regex[4], text):
        return True
    return False

def post_process(tag, decode, df, start_pred, df_now, tokenizer):
    # TODO!!!!!

    if len(df) > 1:
        print('multiple matches')
        ########### TODO:
        total = 0

        start_pred -= (len(tag)+3)
        for i, text in enumerate(df_now['tokenized_text']):
            total += len(tokenizer.encode(text, add_special_tokens=False))
            if total > start_pred:

                break

        df = df_now.iloc[i]

    else:
        df = df.squeeze()
    ID = str(df['doc_id']) + '-' + str(df['line'])

    if len(decode) > 40:
        return ID, 'NONE'

    if tag == all_tag[0]:
        return None, 'NONE'

    if tag == all_tag[1]:

        return ID, 'NONE'

    if tag == all_tag[2]:
        if df['line'] > 15:
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[3]:

        if df['line'] < 10 or df['line'] > 22:
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[4]:
        if df['line'] > 25:
            return ID, 'NONE'

        for c in city:
            if c in decode:
                decode += ' 都道府県:' + c
                return ID, decode

        return ID, 'NONE'

    if tag == all_tag[5]:
        BeInDateTimeFormat(decode)
        year_pos = decode.find('年')
        if year_pos != -1:
            decode += ' 調達年度:' + decode[:year_pos+1]

        if df['line'] > 20:
            decode = 'NONE'
        return ID, decode

    if tag == all_tag[6]:
        if df['line'] > 20:
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[7]:
        if df['line'] > 10:
            decode = 'NONE'

        return ID, decode


    if tag == all_tag[8]:
        if df['line'] < 40:
            decode = 'NONE'

        if not BeInDateTimeFormat(decode):
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[9]:
        return None, 'NONE'
        if df['line'] < 25:
            decode = 'NONE'
        if not BeInDateTimeFormat(decode):
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[10]:
        if df['line'] < 20:
            decode = 'NONE'
        if not BeInDateTimeFormat(decode):
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[11]:
        if df['line'] < 25:
            decode = 'NONE'
        if not BeInDateTimeFormat(decode):
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[12]:
        if df['line'] < 25:
            decode = 'NONE'
        if not BeInDateTimeFormat(decode):
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[13]:
        if df['line'] < 25:
            decode = 'NONE'

        return ID, decode

    if BeInDateTimeFormat(decode):
        return None, 'NONE'

    if tag == all_tag[14]:
        # return None, 'NONE'
        if df['line'] < 15:
            decode = 'NONE'
            return ID, decode

        if not BeInTelTaxFormat(decode):

            print('tel not in format!!!!!!!!', decode)
            decode = 'NONE'
            

        return ID, decode



    if tag == all_tag[15]:
        if df['line'] < 25:
            decode = 'NONE'
            return ID, decode

        return ID, decode

    if tag == all_tag[16]:
        if df['line'] < 25:
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[17]:
        if df['line'] < 40:
            decode = 'NONE'
            return ID, decode

        return ID, decode

    if tag == all_tag[18]:
        if df['line'] < 25:
            decode = 'NONE'

        return ID, decode

    if tag == all_tag[19]:
        if df['line'] < 25:
            decode = 'NONE'

        return ID, decode

def getEnd(start, line_span):
    for span in reversed(line_span):
        if start >= span[0] and start <= span[1]:
            return span[1]


def predict(test_data_set, model, cus_model, device, tokenizer, args):

    
    cus_model.load_state_dict(torch.load(args.load_model, map_location='cuda:0'))
    cus_model.to(device)
    model.eval()
    cus_model.eval()
    dataloader = DataLoader(test_data_set, batch_size=1, collate_fn=create_mini_batch)
    clean_df = pd.read_csv('./preprocess-test-data/clean_data_frame.csv')
    file_list = clean_df['doc_id'].unique()

    print('file_list', file_list)

    if args.test:
        predict_df = pd.read_csv('./test/sample_submission.csv')
    if args.valid:
        predict_df = pd.read_csv('./dev/dev_ref.csv')
    predict_df['Prediction'] = 'NONE'

    with torch.no_grad():

        # Record paragraph and document id we are predicting
        paragraph_now = 1         
        doc_iter = iter(file_list)
        doc_now = next(doc_iter)

        for step, data in tqdm(enumerate(dataloader, 0)):

            paragraph_total = len(clean_df[clean_df['doc_id']==doc_now]['paragraph'].unique())
            line_total = len(clean_df[clean_df['doc_id']==doc_now]['line'].unique())

            token_tensors, token_type_tensors, mask_tensors = [t.to(device) for t in data if t is not None]

            tag = all_tag[step % 20]
            df_now = clean_df.loc[(clean_df['doc_id']==doc_now) & (clean_df['paragraph']==paragraph_now)]

            
            if step % 20 == 0:
                line_span = []
                start = len(tag) + 3
                end = len(tag) + 3
                for row in df_now['tokenized_text']:

                    word = len(tokenizer.tokenize(row))
                    end += (word)
                    line_span.append([start, end])
                    start = end


            outputs = cus_model(input_ids=token_tensors.long(), 
                            token_type_ids=token_type_tensors.long(), 
                            attention_mask=mask_tensors.long(),
                            labels=None)

            softmax = nn.Softmax(dim=-1)
            start_logits = outputs[0].view(-1)

            #start_pred = softmax(start_logits).argmax().item()
            #end_pred = softmax(end_logits).argmax().item()

            

            #####################################################################################################
            # 處理同一個tag在同一段出現兩次的情況
            sort = start_logits.sort(descending=True)
            prob = sort[0].sigmoid()
            index = sort[1]

            start_list = (prob > 0.9).float()
            start_candidate = index[:start_list.tolist().count(1)].tolist()
            end_candidate = []

            remove = []
            for i, start in enumerate(start_candidate, 0):

                if not start:
                    break
                try:
                    end = getEnd(start, line_span)
                    end_logits = outputs[1].view(-1)[start:end]
                    if not end_logits.tolist():
                        remove.append(start)
                        continue
                    end_idx = end_logits.sigmoid().argmax().item()
                    prob = end_logits.sigmoid()[end_idx]
                except:
                    pdb.set_trace() 

                if prob > 0.9:
                    end = end_idx + start
                    end_candidate.append(end)
                else:
                    remove.append(start)

            for out in remove:
                start_candidate.remove(out)


            if not len(start_candidate) == len(end_candidate):
                pdb.set_trace()

            for start_pred, end_pred in zip(start_candidate, end_candidate):
                if not start_candidate:
                    continue

                decode = normalize_text(tokenizer.decode(token_tensors[0][start_pred:end_pred+1]))

            #####################################################################################################

                try:
                    # Find the target exist which line
                    target_df = df_now[df_now['tokenized_text'].str.contains(decode, regex=False, case=False)]
                    regex = ''
                    for char in decode:
                        if char in '()$+-*/%?':
                            char = '\\' + char
                        regex += char

                    span = re.search(regex, target_df['tokenized_text'].values[0], re.IGNORECASE).span()
                    text = target_df['clean_text'].values[0][span[0]:span[1]]
                    raw = target_df['raw_text'].values[0]

                    # deal with half and full 
                    def match(raw, text):
                        raw_half =  unicodedata.normalize('NFKC', raw)
                        
                        regex = ''
                        for char in text:
                            if char in '()$+-*/%?':
                                char = '\\' + char
                            regex += char + '\s*'

                        raw_span = re.search(regex, raw_half).span()
                        return raw[raw_span[0]:raw_span[1]]

                    text = match(raw, text)

                except Exception as e:
                    text = ''
                    # TODO
                    # print('{}: {}\n'.format(e, decode))

                # Produce prediction
                

                text = text.replace(' ', '')
                # print('{} : {}'.format(tag, decode))
                # pdb.set_trace()

                # if doc_now in file_list[4:]:
                #     if step % 20 == 3:
                #         print(tag, decode, paragraph_now, paragraph_total)
                #         # print('start prob: {}, end prob: {}'.format())
                #         pdb.set_trace()
                

                
                if target_df.empty or not text: # Cannot find or text is empty
                    text = 'NONE'
                    line_id = None
                else:
                    line_id, text = post_process(tag, text, target_df, start_pred, df_now, tokenizer)

                # Prediction Fall on the range of tag or paragraph segment and too long text
                if start_pred < len(tag) + 4 or end_pred < len(tag) + 4 or len(text) > 40:
                    text = 'NONE'

                if line_id and line_total:
                    if line_total - int(line_id.split('-')[1]) < 40:
                        text = 'NONE'

                ######################################################################################################
                # TODO: need to convert the original text
                if text == 'NONE':
                    pass

                else:
                    try:
                        if predict_df.loc[predict_df['ID']==line_id, 'Prediction'].item() == 'NONE':
                            predict_df.loc[predict_df['ID']==line_id, 'Prediction'] = tag + ':' + text # How pandas update row value in specific column
                        else:
                            predict_df.loc[predict_df['ID']==line_id, 'Prediction'] = predict_df.loc[predict_df['ID']==line_id, 'Prediction'].item() + ' ' + tag + ':' + text
                    except Exception as e:
                        print(e)

            ########################################################################################################


            # 20 is length of tag
            if step % 20 == 19 and step:
                paragraph_now += 1
                if paragraph_now == paragraph_total + 1:
                    try:
                        doc_now = next(doc_iter)
                    except StopIteration as s:
                        print(s)
                        break
                    paragraph_now = 1

    predict_df.to_csv(args.output_path, index=False)

    return




def predict_crf(test_data_set, crf_model, cus_model, device, tokenizer, args):

    crf_model.load_state_dict(torch.load('./model/bert_bilstm_crf.pth', map_location='cuda:0'))
    crf_model.to(device)
    cus_model.load_state_dict(torch.load(args.load_model, map_location='cuda:0'))
    cus_model.to(device)
    crf_model.eval()
    cus_model.eval()
    dataloader = DataLoader(test_data_set, batch_size=1, collate_fn=create_mini_batch)
    clean_df = pd.read_csv('./preprocess-test-data/clean_data_frame.csv')
    file_list = clean_df['doc_id'].unique()

    print('file_list', file_list)

    if args.test:
        predict_df = pd.read_csv('./test/sample_submission.csv')
    if args.valid:
        predict_df = pd.read_csv('./dev/dev_ref.csv')
    predict_df['Prediction'] = 'NONE'
    predict_df['have_tag'] = False

    # use crf to check tag existness
    crf_bool = [] 



    with torch.no_grad():

        # Record paragraph and document id we are predicting
        paragraph_now = 1
        doc_iter = iter(file_list)
        doc_now = next(doc_iter)

        for step, data in enumerate(tqdm(dataloader), 0):

            paragraph_total = len(clean_df[clean_df['doc_id']==doc_now]['paragraph'].unique())
            line_total = len(clean_df[clean_df['doc_id']==doc_now]['line'].unique())
            tag = all_tag[step % 20]
            df_now = clean_df.loc[(clean_df['doc_id']==doc_now) & (clean_df['paragraph']==paragraph_now)]



            token_tensors, token_type_tensors, mask_tensors = [t.to(device) for t in data if t is not None]

            if step % 20 == 0:
                _, y_hat = crf_model(token_tensors.long())

                start = len(tag) + 3
                end = len(tag) + 3
                for row in df_now['tokenized_text']:
                    word = len(tokenizer.tokenize(row))
                    end += word
                    crf_tag = y_hat[0][start:end].tolist()
                    have_tag = True if crf_tag.count(3)/len(crf_tag) < 0.5 else False
                    crf_bool.append(have_tag)
                    
                    start = end


            outputs = cus_model(input_ids=token_tensors.long(), 
                            token_type_ids=token_type_tensors.long(), 
                            attention_mask=mask_tensors.long(),
                            labels=None)

            softmax = nn.Softmax(dim=-1)
            start_logits = outputs[0].view(-1)
            end_logits = outputs[1].view(-1)

            
            pred_tags = []
            for t in y_hat.squeeze():
                pred_tags.append(idx2tag[t.item()])

            #start_pred = softmax(start_logits).argmax().item()
            #end_pred = softmax(end_logits).argmax().item()

            #####################################################################################################
            # 處理同一個tag在同一段出現兩次的情況
            sort = start_logits.sort(descending=True)
            prob = sort[0].sigmoid()
            index = sort[1]

            start_list = (prob > 0.9).float()
            start_candidate = index[:start_list.tolist().count(1)]

            sort = end_logits.sort(descending=True)
            prob = sort[0].sigmoid()
            index = sort[1]

            end_list = (prob > 0.9).float()
            end_candidate = index[:end_list.tolist().count(1)]



            for start_pred, end_pred in zip(start_candidate, end_candidate):
                decode = normalize_text(tokenizer.decode(token_tensors[0][start_pred:end_pred+1])) 

            #####################################################################################################

                try:
                    # Find the target exist which line
                    target_df = df_now[df_now['tokenized_text'].str.contains(decode, regex=False, case=False)]
                    regex = ''
                    for char in decode:
                        if char in '()$+-*/%?':
                            char = '\\' + char
                        regex += char

                    span = re.search(regex, target_df['tokenized_text'].values[0], re.IGNORECASE).span()
                    text = target_df['clean_text'].values[0][span[0]:span[1]]
                    raw = target_df['raw_text'].values[0]

                    # deal with half and full 
                    def match(raw, text):
                        raw_half =  unicodedata.normalize('NFKC', raw)
                        
                        regex = ''
                        for char in text:
                            if char in '()$+-*/%?':
                                char = '\\' + char
                            regex += char + '\s*'

                        raw_span = re.search(regex, raw_half).span()
                        return raw[raw_span[0]:raw_span[1]]

                    text = match(raw, text)

                except Exception as e:
                    text = ''
                    # TODO
                    # print('{}: {}\n'.format(e, decode))

                # Produce prediction
                

                text = text.replace(' ', '')
                # print('{} : {}'.format(tag, decode))
                # pdb.set_trace()

                # if doc_now in file_list[4:]:
                
                
                if target_df.empty or not text: # Cannot find or text is empty
                    text = 'NONE'
                    line_id = None
                else:
                    line_id, text = post_process(tag, text, target_df, start_pred, df_now, tokenizer)

                # Prediction Fall on the range of tag or paragraph segment and too long text
                if start_pred < len(tag) + 4 or end_pred < len(tag) + 4 or len(text) > 40:
                    text = 'NONE'

                if line_id and line_total:
                    if line_total - int(line_id.split('-')[1]) < 40:
                        text = 'NONE'

                if step % 20 == 14:
                    print(tag, decode, paragraph_now, paragraph_total)
                    # print('start prob: {}, end prob: {}'.format())

                ######################################################################################################
                if text == 'NONE':
                    pass

                else:
                    try:
                        if predict_df.loc[predict_df['ID']==line_id, 'Prediction'].item() == 'NONE':
                            predict_df.loc[predict_df['ID']==line_id, 'Prediction'] = tag + ':' + text # How pandas update row value in specific column
                        else:
                            predict_df.loc[predict_df['ID']==line_id, 'Prediction'] = predict_df.loc[predict_df['ID']==line_id, 'Prediction'].item() + ' ' + tag + ':' + text
                    except Exception as e:
                        print(e)

            ########################################################################################################


            # 20 is length of tag
            if step % 20 == 19 and step:
                paragraph_now += 1
                if paragraph_now == paragraph_total + 1:
                    try:
                        doc_now = next(doc_iter)
                    except StopIteration as s:
                        print(s)
                        break
                    paragraph_now = 1

    predict_df['have_tag'] = crf_bool
    #predict_df.loc[predict_df.have_tag==False, 'Prediction']='NONE'
    del predict_df['have_tag']
    #pdb.set_trace()
    predict_df.to_csv(args.output_path, index=False)

    return



def main():
    args = parse()
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
    model, cus_model, device = setting(model_version, args)

    # Preprocess
    raw_data = data_to_paragraph(args, tokenizer)
    tag_paragraph = tag_to_paragraph(raw_data)
    input_ids, token_type, mask_attention = tokenization(tokenizer, tag_paragraph)
    np.save('./preprocess-test-data/input_ids.npy', input_ids)
    np.save('./preprocess-test-data/token_type.npy', token_type)
    np.save('./preprocess-test-data/mask_attention', mask_attention)

    test_data_set = EE_dataset(mode='test')
    if args.bert:
        predict(test_data_set, model, cus_model, device, tokenizer, args)
    elif args.bert_bilstm_crf:
        predict_crf(test_data_set, model, cus_model, device, tokenizer, args)

if __name__ == '__main__':
    main()