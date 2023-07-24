#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model import CLForClassification

def load_data(para_list):
    test_data = []
    for id, para in enumerate(para_list):
        if id == 0:
            para_pre = para
            continue
        test_data.append((para_pre, para))
        para_pre = para
    return test_data

def dict_to_device(dic, device):
    new_dict = {}
    for k, v in dic.items():
        new_dict[k] = v.to(device)
    return new_dict

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        text_1 = self.data[index][0]
        text_2 = self.data[index][1]
        return self.text_2_id(text_1), self.text_2_id(text_2)


class Predictor:
    def __init__(self, pretrain_model, finetune, device, batch_size=16, max_length=512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        self.model = CLForClassification(pretrained_model=pretrain_model, num_labels=2)
        self.device = device
        if self.device == "cpu":
            self.model.load_state_dict(torch.load(finetune, map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load(finetune))
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.to(self.device)

    def run_predict(self, para_list):
        test_data = load_data(para_list)
        test_dataloader = DataLoader(TestDataset(test_data, self.tokenizer, self.max_length), batch_size=self.batch_size)
        self.model.eval()
        y_pred_array = np.array([])
        with torch.no_grad():
            for source, target in test_dataloader:
                source = dict_to_device(source, self.device)
                target = dict_to_device(target, self.device)
                loss, logits = self.model(source, target)
                y_pred = np.argmax(logits.cpu(),axis=1).numpy().squeeze()
                y_pred_array = np.append(y_pred_array, y_pred)
        return y_pred_array.astype(int).tolist()

if __name__ == '__main__':
    corpus = ["I'm not arguing with you here, I'm simply trying to contextualize this for you. To the extent that they are there, it is with your consent. The state has passed laws making sure that vulnerable people (not saying he's one) don't get abused (not saying you're abusing him), and in casting a wide net to save as many vulnerable little birds as possible from hitting the floor after being kicked out of their nest wrongfully, the state has (as much from a lack of better options as from any other reason) created a circumstance where occasionally some not-so-vulnerable little bird can take advantage of someone else's nest.", 
              "He's at my place half the time and his fiance(e)'s place the other half of the time. He's been (homeless) couch surfing for several years and only recently got engaged to his other partner. We don't have any current issues that would lead me to want this arrangement to stop, but I do want to protect my own legal rights.",
              "I don't think he, in particular, would do that, but I do want to retain my own legal rights wherever possible/appropriate."
              ]
    p = Predictor("../pretrain/roberta-base", "./classifier1/f1_0.7890.pt", "cuda:0")
    pred = p.run_predict(corpus)
    print(pred)