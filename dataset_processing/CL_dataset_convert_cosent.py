#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import random
from tqdm import tqdm
from posi_nega_data_builder import PositiveNegativeBuilder

random.seed(42)
class CLDatasetConvertNaive:
    def __init__(self, file_path, file_type):
        self.file_type = file_type
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        print('loading ' + self.file_type + ' data')
        
        labels = {}
        for label in glob.glob(os.path.join(file_path, 'truth-problem-*.json')):
            with open(label, 'r', encoding='utf-8') as lf:
                curr_label = json.load(lf)
                labels[os.path.basename(label)[14:-5]] = curr_label

        loaded_data = []
        for idx, document_path in enumerate(tqdm(glob.glob(file_path + '/*.txt'))):
            with open(document_path, encoding="utf8") as file:
                document = file.read()
            share_id = os.path.basename(document_path)[8:-4]
            para_list = document.split('\n')
            curr_label = labels[share_id]
            authors = curr_label["authors"]
            changes_list = curr_label["changes"]
            try:
                PN_builder = PositiveNegativeBuilder(para_list, changes_list, authors)
            except Exception:
                print(f"[E] {share_id}")
            if self.file_type == 'train':
                corpus_data = []
                corpus_data.append(PN_builder.build_positive_instances())
                if sum(changes_list) + 1 == authors:
                    corpus_data.append(PN_builder.build_negative_transparent())
                else:
                    corpus_data.append(PN_builder.build_negative_instances())
                random.shuffle(corpus_data)
                loaded_data.extend(corpus_data)
            elif self.file_type == 'valid':
                loaded_data.append(PN_builder.build_vaild_instances())
        return loaded_data
    
    def convert(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print('converting ' + self.file_type + ' data')
        pos_num = 0
        neg_num = 0
        if self.file_type == 'train':
            with open(f"{out_path}/{self.file_type}.txt", 'a', encoding='utf-8') as fp:
                for corpus in tqdm(self.data):
                    for instances in corpus:
                        fp.write(json.dumps({"para_a": instances[0], "para_b": instances[1], "label": instances[2]}))
                        fp.write("\n")
                        if int(instances[2]) == 0:
                            pos_num += 1
                        else:
                            neg_num += 1

        elif self.file_type == 'valid':
            with open(f"{out_path}/{self.file_type}.txt", 'a', encoding='utf-8') as fp:
                for corpus in tqdm(self.data):
                    for instances in corpus:
                        fp.write(json.dumps({"para_a": instances[0], "para_b": instances[1], "label": instances[2]}))
                        fp.write("\n")
                        if int(instances[2]) == 0:
                            pos_num += 1
                        else:
                            neg_num += 1
        print(f"pos_num: {pos_num}, neg_num: {neg_num}")

class CLDatasetConvertAdvanced:
    pass

if __name__ == "__main__":
    for task in range(1,4):
        for data_type in ["train", "valid"]:
            pan_path = f"./pan23_dataset/dataset{task}/{data_type}"
            cl_path = f"./cl_dataset_cosent_transparent/dataset{task}"
            cl_data = CLDatasetConvertNaive(pan_path, data_type)
            cl_data.convert(cl_path)
