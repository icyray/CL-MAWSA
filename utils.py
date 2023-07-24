#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)


class EncoderDataset(BaseDataset):
    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        text_1 = self.data[index]["para_a"]
        text_2 = self.data[index]["para_b"]
        label_temp = self.data[index]["label"]
        label = 0 if int(label_temp) == 1 else 1 # reverse the label from pan official, which match the result of cos_sim
        return self.text_2_id([text_1, text_2]), np.array(label)


class ClassifierDataset(BaseDataset):
    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        text_1 = self.data[index]["para_a"]
        text_2 = self.data[index]["para_b"]
        label = int(self.data[index]["label"])
        return self.text_2_id(text_1), self.text_2_id(text_2), np.array(label)
