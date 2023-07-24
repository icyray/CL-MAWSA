#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import json
import torch
from tqdm import tqdm

from predict import Predictor


def main(args):
    os.makedirs(args.output, exist_ok=True)
    p = Predictor(args.pretrain_model, args.finetune_model, args.device, args.batch_size)
    for idx, document_path in enumerate(tqdm(glob.glob(args.input + '/*.txt'))):
        share_id = os.path.basename(document_path)[8:-4]
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        para_list = document.split('\n')
        pred = {"changes": p.run_predict(para_list)}
        with open(os.path.join(args.output, f"solution-problem-{share_id}.json"),'w',encoding="utf8") as fw:
            json.dump(pred,fw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="../pan23_dataset/test", type=str)
    parser.add_argument("-o", "--output", default="../pan23_dataset/test", type=str)
    parser.add_argument("--pretrain_model", default="../pretrain/roberta-base", type=str)
    parser.add_argument("--finetune_model", default="./classifier/best.pt", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    main(args)