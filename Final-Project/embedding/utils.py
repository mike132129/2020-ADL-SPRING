import unicodedata, re
import torch
from transformers import BertModel, BertForMaskedLM
def setting(model_version):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = BertForMaskedLM.from_pretrained(model_version)
    model.to(device)

    return model, device

def normalize_text(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag
