from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import pdb
import random

class modified_bert(nn.Module):
    def __init__(self, model):
        super(modified_bert, self).__init__()
        self.model = model
        self.qa_output = nn.Linear(768, 2)
        self.cls_output = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        bert_output = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
 
        sequence_output = bert_output[0]
        cls_output = bert_output[1]

        qa_logits = self.qa_output(sequence_output)
        cls_logits = self.cls_output(cls_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        
        
        # Squeeze
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        cls_logits = cls_logits.squeeze(-1)

        if labels == None: # when predicting
            return start_logits, end_logits, cls_logits
        

        cls_target = labels[:, 0]
        start_position = labels[:, 1]
        end_position = labels[:, 2]

        # the start or end position outside model inputs, we ignore these terms
        ignore_index = 0

        for i, n in enumerate(start_position):
            if torch.isnan(n):
                start_position[i] = ignore_index
                end_position[i] = ignore_index

        loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        start_loss = loss_fct(start_logits, start_position.long())
        end_loss = loss_fct(end_logits, end_position.long())

        bce_loss = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        cls_loss = bce_loss(sigmoid(cls_logits), cls_target.float())

        output = ([start_loss, end_loss, cls_loss], cls_logits, start_logits, end_logits) 
        return output











