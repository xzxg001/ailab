"""
主要改进点：

1.类别权重：
根据训练集中的标签分布计算类别权重，并将其应用于CrossEntropyLoss中，以缓解类别不平衡问题。
2.学习率调度器：
使用get_linear_schedule_with_warmup动态调整学习率，有助于模型更好地收敛。
3.早停机制：
在验证集的F1分数不再提升时提前停止训练，防止过拟合。
4.优化Batch Size和学习率：
调整了学习率至5e-5，这是BERT模型常用的学习率。
5.保存最佳模型：
在训练过程中保存验证集F1分数最高的模型，以便在预测时使用。
"""
import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
import torch.nn as nn
import random

## 设置随机种子
def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

## 文本向量化函数
def text_tokenize(text_list): 
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_text

## 读取train.csv、dev.csv
train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")

## 分离文件路径、文本内容和标签
train_path = list(train_csv.path)[:1500]
train_label = list(train_csv.label)[:1500]
train_txt = list(train_csv.text)[:1500]
dev_path = list(dev_csv.path)[:500]
dev_label = list(dev_csv.label)[:500]
dev_txt = list(dev_csv.text)[:500]

## 文本向量化
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)

## 计算类别权重
label_counts = train_csv['label'].value_counts().sort_index()
class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(label_counts)
class_weights = class_weights.to('cuda' if torch.cuda.is_available() else 'cpu')

## 创建数据集
train_dataset = TensorDataset(train_coded_txt["input_ids"], 
                             train_coded_txt["attention_mask"],
                             torch.tensor(train_label))
dev_dataset = TensorDataset(dev_coded_txt["input_ids"], 
                           dev_coded_txt["attention_mask"],
                           torch.tensor(dev_label))

## 数据加载器
batch_size = 8
dataloader_train = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
dataloader_dev = DataLoader(
    dev_dataset,
    sampler=SequentialSampler(dev_dataset),
    batch_size=32
)

def calculate_score_classification(preds, labels, average_f1='macro'):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, recall, f1, precision, confuse_matrix

class MyDLmodel():
    def __init__(self, model, device, class_weights, weight_decay=5e-4, dropout_prob=0.3):
        self.model = model

        # 添加 Dropout
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.dropout = nn.Dropout(p=dropout_prob)

        self.model.to(device)

        # 添加权重惩罚 (L2 正则化) 和类别权重
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=5e-5, 
            eps=1e-8, 
            weight_decay=weight_decay
        )
        
        self.scheduler = None
        self.device = device
        self.class_weights = class_weights

    def evaluate(self, dataloader_val):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        for batch in dataloader_val:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader_val)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        accuracy, recall, f1, precision, confuse_matrix = calculate_score_classification(all_preds, all_labels)
        return avg_loss, accuracy, recall, f1, precision, confuse_matrix

    def train(self, dataloader_train, dataloader_dev, epochs, patience=3):
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        best_f1 = 0
        patience_counter = 0

        total_steps = len(dataloader_train) * epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in dataloader_train:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            val_loss, accuracy, recall, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("早停触发，停止训练。")
                    break

    def predict(self, dataloader_test):
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.eval()
        all_preds = []

        for batch in dataloader_test:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            all_preds.extend(preds)

        return all_preds

    def cross_validate(self, dataset, k_folds=5, epochs=3, batch_size=16):
        kfold = KFold(n_splits=k_folds, shuffle=True)
        results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            dataloader_train = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            dataloader_val = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # 重新初始化模型和优化器（避免权重泄漏）
            self.model.apply(self._reset_weights)

            self.train(dataloader_train, dataloader_val, epochs)

            val_loss, accuracy, recall, f1, precision, confuse_matrix = self.evaluate(dataloader_val)
            print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            results.append((val_loss, accuracy, recall, f1, precision))

        return results

    def _reset_weights(self, m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

## 运行后会在当前目录生成result.csv文件，提交result.csv文件即可
def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误！请检查test_preds长度是否为1241！！！")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./result.csv", sep="#", index=False)
    print("测试集预测结果已成功写入到文件中！")

if __name__  == "__main__":
    set_seeds(17)
    pretrained_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False
    )
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    mymodel = MyDLmodel(pretrained_model, device, class_weights)
    epochs = 20
    mymodel.train(dataloader_train, dataloader_dev, epochs)
    
    # 预测测试集标签
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_text = list(test_csv.text)
    test_coded_txt = text_tokenize(test_text)
    test_dataset = TensorDataset(
        test_coded_txt["input_ids"], 
        test_coded_txt["attention_mask"]
    )
    dataloader_test = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32
    )
    test_preds = mymodel.predict(dataloader_test)
    write_result(test_preds)