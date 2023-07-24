#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import ast

from loguru import logger
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import CLModel, CLForClassification, cosent_loss
from utils import EncoderDataset, ClassifierDataset, load_data


def seed_everything(seed=42):
    '''
    Set the seed for the environment
    :param seed: int
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def train_encoder(args, model, train_dataloader, eval_dataloader, optimizer, scheduler):
    '''
    Encoder Trainer
    '''
    best = 0
    step = 0
    early_stop_batch = 0
    train_loss = 0.

    for epoch in range(args.train_epochs):
        logger.info(f"Epoch [{epoch+1}/{args.train_epochs}]")
        model.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader), start=1):
            batch, labels = data
            labels =  labels.to(args.device)
            real_batch_num = batch.get('input_ids').shape[0]
            input_ids = batch.get('input_ids').view(real_batch_num * 2, -1).to(args.device)
            attention_mask = batch.get('attention_mask').view(real_batch_num * 2, -1).to(args.device)

            # input_ids = batch.get('input_ids').to(args.device)
            # attention_mask = batch.get('attention_mask').to(args.device)

            # Train
            out = model(input_ids, attention_mask)
            loss = cosent_loss(labels, out, args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            step += 1
            # Eval
            if batch_idx % args.culcalate_step == 0:
                model.eval()
                corrcoef, train_f1 = eval_encoder(model, eval_dataloader, args.device)
                logger.info('Step: {}, loss: {:.4f}, best:{:.4f}, train corrcoef:{:.4f}, train_f1:{:.4f}'.format(step, train_loss / args.culcalate_step, best, corrcoef, train_f1))
                train_loss = 0
                model.train()
                if best < corrcoef:
                    early_stop_batch = 0
                    best = corrcoef
                    torch.save(model.state_dict(), os.path.join(args.save_path, "corrcoef_{:.4f}.pt".format(best)))
                    logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                    continue
                early_stop_batch += 1
                if args.early_stop != 0:
                    if early_stop_batch == args.early_stop:
                        logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                        logger.info(f"train use sample number: {(batch_idx - 10) * args.train_batch_size}")
                        return
        logger.info("saving checkpoint...")
        torch.save(model.state_dict(), os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}.pt"))
        model.eval()
        corrcoef, train_f1 = eval_encoder(model, eval_dataloader, args.device)
        logger.info(f"epoch {epoch+1} corrcoef: {corrcoef:.4f}, train_f1: {train_f1:.4f}")

def train_classifier(args, model, train_dataloader, eval_dataloader, optimizer, scheduler):
    '''
    Classifier Trainer
    '''
    best = 0
    step = 0
    early_stop_batch = 0
    train_loss = 0.

    for epoch in range(args.train_epochs):
        logger.info(f"Epoch [{epoch+1}/{args.train_epochs}]")
        model.train()
        for batch_idx, data in enumerate(tqdm(train_dataloader), start=1):
            source, target, label = data
            source = dict_to_device(source, args.device)
            target = dict_to_device(target, args.device)
            label =  label.to(args.device)
            # Train
            loss, logits = model(source, target, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            step += 1
            # Eval
            if batch_idx % args.culcalate_step == 0:
                model.eval()
                train_acc, train_p, train_r, train_f1 = eval_classifier(model, eval_dataloader, args.device)
                logger.info(f'Step: {step}, loss: {(train_loss / args.culcalate_step):.4f}, f1:{train_f1:.4f}, acc:{train_acc:.4f}, precision:{train_p:.4f}, recall:{train_r:.4f}')
                train_loss = 0
                model.train()
                if best < train_f1:
                    early_stop_batch = 0
                    best = train_f1
                    torch.save(model.state_dict(), os.path.join(args.save_path, "f1_{:.4f}.pt".format(best)))
                    logger.info(f"higher f1: {best:.4f} in batch: {batch_idx}, save model")
                    continue
                early_stop_batch += 1
                if args.early_stop != 0:
                    if early_stop_batch == args.early_stop:
                        logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                        logger.info(f"train use sample number: {(batch_idx - 10) * args.train_batch_size}")
                        return
        logger.info("saving checkpoint...")
        torch.save(model.state_dict(), os.path.join(args.save_path, f"checkpoint_epoch_{epoch+1}.pt"))
        model.eval()
        train_acc, train_p, train_r, train_f1 = eval_classifier(model, eval_dataloader, args.device)
        logger.info(f'epoch {epoch+1}: f1:{train_f1:.4f}, acc:{train_acc:.4f}, precision:{train_p:.4f}, recall:{train_r:.4f}')


def eval_encoder(model, dataloader, device) -> float:
    '''
    Evaluation function for Encoder
    It calculate cos_sim and corresponding spearman correlation
    Correlation closer to 1 means better
    '''
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for batch, labels in tqdm(dataloader, total=len(dataloader)):

            real_batch_num = batch.get('input_ids').shape[0]
            input_ids = batch.get('input_ids').view(real_batch_num * 2, -1).to(args.device)
            attention_mask = batch.get('attention_mask').view(real_batch_num * 2, -1).to(args.device)

            output = model(input_ids, attention_mask)

            source_pred = output[::2]
            target_pred = output[1::2]
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(labels.cpu().numpy()))

        sim_tensor = sim_tensor.cpu().numpy()

        corrcoef = spearmanr(label_array, sim_tensor).correlation
        best_f1, best_precision, best_recall, threshold = find_best_f1_and_threshold(sim_tensor, label_array)

    return corrcoef, best_f1

def find_best_f1_and_threshold(scores, labels, high_score_more_similar=True):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows)-1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold

def eval_classifier(model, dataloader, device) -> float:
    '''
    Evaluation function for Classifier
    '''
    model.eval()
    y_pred_array = np.array([])
    label_array = np.array([])
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            source, target, label = data
            source = dict_to_device(source, device)
            target = dict_to_device(target, device)
            label_array = np.append(label_array, np.array(label))
            label =  label.to(device)
            loss, logits = model(source, target, label)
            y_pred = np.argmax(logits.cpu(),axis=1).numpy().squeeze()
            y_pred_array = np.append(y_pred_array, y_pred)
            

    train_acc = metrics.accuracy_score(label_array, y_pred_array)
    train_p = metrics.precision_score(label_array, y_pred_array)
    train_r = metrics.recall_score(label_array, y_pred_array)
    train_f1 = metrics.f1_score(label_array, y_pred_array)

    return train_acc, train_p, train_r, train_f1

def dict_to_device(dic, device):
    new_dict = {}
    for k, v in dic.items():
        new_dict[k] = v.to(device)
    return new_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_task", nargs="?", choices=["encoder", "classifier"], type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--task_name", default="subtask", type=str)
    parser.add_argument("--train_file", default="../cl_dataset/lite/train.txt", type=str)
    parser.add_argument("--test_file", default="../cl_dataset/lite/valid.txt", type=str)
    parser.add_argument("--shuffle_train", default=True, type=ast.literal_eval)
    parser.add_argument("--early_stop", default=0, type=int)
    
    parser.add_argument("--save_path", default="./save_model", type=str)
    parser.add_argument("--pretrain_model", default="../pretrain/roberta-base", type=str)
    parser.add_argument("--finetune_model", type=str)
    parser.add_argument("--pooler_type", default="cls", choices=["cls", "pooler", "last-avg", "first-last-avg"], type=str)

    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)
    parser.add_argument("--culcalate_step", default=500, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    assert(args.train_task is not None)
    seed_everything(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.task_name), exist_ok=True)
    args.save_path = os.path.join(args.save_path, args.task_name)
    logger.add(os.path.join(args.save_path, f"{args.task_name}.log"))
    logger.info(args)

    train_data = load_data(args.train_file)
    eval_data = load_data(args.test_file)
    if args.shuffle_train:
        random.shuffle(train_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)

    if args.train_task == "encoder":
        logger.info("Training encoder")
        train_dataloader = DataLoader(EncoderDataset(train_data, tokenizer, args.max_length), batch_size=args.train_batch_size)
        eval_dataloader = DataLoader(EncoderDataset(eval_data, tokenizer, args.max_length), batch_size=args.train_batch_size)
        model = CLModel(args.pretrain_model, args.pooler_type)
        train = train_encoder

    elif args.train_task == "classifier":
        assert(args.finetune_model is not None)
        logger.info("Training classifier")
        train_dataloader = DataLoader(ClassifierDataset(train_data, tokenizer, args.max_length), batch_size=args.train_batch_size)
        eval_dataloader = DataLoader(ClassifierDataset(eval_data, tokenizer, args.max_length), batch_size=args.train_batch_size)
        model = CLForClassification(pretrained_model=args.pretrain_model, pooler_type=args.pooler_type, finetune_model=args.finetune_model, num_labels=2)
        train = train_classifier

    model.to(args.device)
    total_steps = len(train_dataloader) * args.train_epochs
    logger.info(f"total steps: {total_steps}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=args.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    train(args, model, train_dataloader, eval_dataloader, optimizer, scheduler)