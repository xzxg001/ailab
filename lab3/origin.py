import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as npW
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
## 文本向量化函数
## text_list 文本内容的list
## 返回一个字典，{'input_ids':value, 'token_type_ids':value, 'attention_mask':value},每个元素的长度等于len(text_list)
## "input_ids"-词转换为数字后的序列 'token_type_ids'-标记一段文本中不同句子的序号 'attention_mask'-标记填充位置的序号 
## reference: https://huggingface.co/docs/transformers/main/en/glossary
def text_tokenize(text_list): 
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased',do_lower_case=True)
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_text
## 'input_ids' 'token_type_ids' 'attention_mask'
## 读取train.csv、dev.csv
train_csv = pd.read_csv("./CSVfile/train.csv", sep = "#")
dev_csv = pd.read_csv("./CSVfile/dev.csv", sep = "#")
## 分离文件路径、文本内容和标签
## 训练时间较长，建议可以先截取部分样本进行代码正确性验证，再使用全部样本
# train_path = list(train_csv.path)[:1500]
# train_label = list(train_csv.label)[:1500]
# train_txt = list(train_csv.text)[:1500]
# dev_path = list(dev_csv.path)[:500]
# dev_label = list(dev_csv.label)[:500]
# dev_txt = list(dev_csv.text)[:500]

train_path = list(train_csv.path)
train_label = list(train_csv.label)
train_txt = list(train_csv.text)
dev_path = list(dev_csv.path)
dev_label = list(dev_csv.label)
dev_txt = list(dev_csv.text)
##  'input_ids' 'token_type_ids' 'attention_mask'
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)
train_dataset = TensorDataset(train_coded_txt["input_ids"], 
                              train_coded_txt["attention_mask"],
                              torch.tensor(train_label))
dev_dataset = TensorDataset(dev_coded_txt["input_ids"], 
                            dev_coded_txt["attention_mask"],
                            torch.tensor(dev_label))
print(len(train_dataset),len(dev_dataset))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
## 这里的batch_size 可以从1、2、4、8、16...尝试，过大的batch_size会使训练过程因为显存不足失败
batch_size = 8
dataloader_train = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
dataloader_dev = DataLoader(
    dev_dataset,
    sampler=RandomSampler(dev_dataset),
    batch_size=32
)
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
def calculate_score_classification(preds, labels, average_f1='macro'):  # weighted, macro
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np

class MyDLmodel():
    def __init__(self, model, device, weight_decay=5e-4, dropout_prob=0.3):
        self.model = model

        # 添加 Dropout
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.dropout = nn.Dropout(p=dropout_prob)

        self.model.to(device)

        # 添加权重惩罚 (L2 正则化)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=5e-6, 
            eps=1e-8, 
            weight_decay=weight_decay
        )
        
        self.scheduler = None
        self.device = device

    def evaluate(self, dataloader_val):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        criterion = nn.CrossEntropyLoss()

        for batch in dataloader_val:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                logits = outputs.logits

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader_val)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        accuracy, ua, f1, precision, confuse_matrix = calculate_score_classification(all_preds, all_labels)
        return avg_loss, accuracy, ua, f1, precision, confuse_matrix

    def train(self, dataloader_train, dataloader_dev, epochs):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in dataloader_train:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Train/Loss', avg_train_loss, epoch+1)
            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Confusion Matrix:\n{confuse_matrix}")
            writer.add_scalar('Validation/Loss', val_loss, epoch+1)
            writer.add_scalar('Validation/Accuracy', accuracy, epoch+1)
            writer.add_scalar('Validation/F1_Score', f1, epoch+1)
            writer.add_scalar('Validation/Precision', precision, epoch+1)

    def predict(self, dataloader_test):
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

            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_val)
            print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            results.append((val_loss, accuracy, ua, f1, precision))

        return results

    def _reset_weights(self, m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
import random
## 设置随机种子
def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
## test_preds 长度为1241的list，对应测试集中1241个样本的标签
##运行后会在当前目录生成result.csv文件，提交result.csv文件即可
##如果没有生成，请检查test_preds的长度是否为1241！
def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误！请检查test_preds长度是否为1241！！！")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv",sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./result.csv",sep = "#")
    print("测试集预测结果已成功写入到文件中！")
from transformers import BertForSequenceClassification
## 特征处理函数，可以对提取的特征进行处理，以获得更好的特征表示
def feature_process(feature):
    return feature
## model reference： https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification
if __name__  == "__main__":
    set_seeds(17)
    writer = SummaryWriter('/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/lab3/log/origin')
    pretrained_model = pretrained_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    mymodel = MyDLmodel(pretrained_model,device)
    epochs = 20
    mymodel.train(dataloader_train,dataloader_dev,epochs)    
    writer.close()