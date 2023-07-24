#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CLModel(nn.Module):
    '''
    Definition of CoSENT
    We only use the CLS as pooling
    '''
    def __init__(self, pretrained_model: str, pooling: str):
        super(CLModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask):
        
        out = self.encoder(input_ids, attention_mask, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
                  
            
def cosent_loss(y_true, y_pred, device):
    '''
    CoSENT loss func
    Ref: https://github.com/shawroad/CoSENT_Pytorch
    '''
    # l2 normalized, so then we just multiply them together to get the cos values.
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    # Multiplication of odd and even vectors
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20


    y_pred = y_pred[:, None] - y_pred[None, :]  # Calculate the difference between the two cosines of all the positions.
    # [y_pred]_ij represents the i_cosine - j_cosine value in the matrix
    y_true = y_true[:, None] < y_true[None, :]   # get the difference of neg - pos
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 0 is added here because e^0 = 1 is equivalent to adding 1 to the log
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0) 
        
    return torch.logsumexp(y_pred, dim=0)

class CLClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class CLForClassification(nn.Module):
    def __init__(self, pretrained_model, num_labels, pooler_type="cls" ,finetune_model=None, pretrain_hidden_size=768):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = CLModel(pretrained_model, pooler_type)
        if finetune_model is not None:
            self.encoder.load_state_dict(torch.load(finetune_model))
        # pretrain_hidden_size = self.encoder.state_dict().get(list(self.encoder.state_dict())[-1]).shape[0]
        self.classifier = CLClassificationHead(pretrain_hidden_size * 3, self.num_labels, 0.1)
    
    def forward(self, source, target, labels=None):
        with torch.no_grad():
            source_out = self.encoder(input_ids=source['input_ids'].squeeze(1), attention_mask=source['attention_mask'].squeeze(1))
            target_out = self.encoder(input_ids=target['input_ids'].squeeze(1), attention_mask=target['attention_mask'].squeeze(1))
            difference = torch.abs(source_out - target_out)
            features = torch.cat((source_out, target_out, difference), 1)

        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits