import unicodedata, re
import torch
from module import modified_bert, Bert_BiLSTM_CRF
from transformers import BertModel
from ohiyo import tag2idx
import pdb
def setting(model_version, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('load model from : {}'.format(model_version))
    if args.bert:
        model = BertModel.from_pretrained(model_version)
        model.to(device)
        cus_model = modified_bert(model)
        cus_model.to(device)

        return model, cus_model, device

    elif args.bert_bilstm_crf:
        model = BertModel.from_pretrained(model_version)
        model.to(device)
        crf_model = Bert_BiLSTM_CRF(tag2idx)
        crf_model.to(device)
        cus_model = modified_bert(model)
        cus_model.to(device)

        return crf_model, cus_model, device

def normalize_text(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag
