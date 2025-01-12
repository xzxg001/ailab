"""
1. 数据增强
由于数据集存在类别不平衡，可以尝试以下方法：

数据增强： 对少数类进行过采样或使用方法生成更多数据，如通过回译或同义词替换生成新的文本样本。
权重调整： 在 CrossEntropyLoss 中通过 weight 参数为少数类赋予更高的权重。
2. 调整学习率和优化器
目前的学习率较低，可以尝试学习率调度器（如余弦退火）或者用不同的学习率试验是否能提升性能。

3. 模型增强
冻结 BERT 的部分层： 仅训练最后几层以提高训练稳定性。
多层感知器 (MLP)： 在分类头部添加额外的全连接层和激活函数（如 ReLU）。
4. K-Fold 交叉验证
使用多折交叉验证提高模型在验证集上的泛化能力。

5. 其他预训练模型
尝试使用更强的预训练模型（如 RoBERTa、DeBERTa）替代 BERT。

6. 微调超参数
尝试调整超参数：

增加 Dropout（如 0.5）。
减少或增加 batch size（如测试 4 或 16）。
调整训练轮次（如耐心等待最佳早停）。
"""
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
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np
import random
# 数据增强（过采样少数类）
from transformers import BertForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import re

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/lab3_dataem')
log_dir = '/old_home/lyt/zxj_workplaces/ailab/lab3/log'
tb_writer = SummaryWriter(log_dir=log_dir)
## 文本向量化函数
## text_list 文本内容的list
## 返回一个字典，{'input_ids':value, 'token_type_ids':value, 'attention_mask':value},每个元素的长度等于len(text_list)
## "input_ids"-词转换为数字后的序列 'token_type_ids'-标记一段文本中不同句子的序号 'attention_mask'-标记填充位置的序号 
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


## 读取train.csv、dev.csv
# train_csv = pd.read_csv("./CSVfile/train.csv", sep = "#")
# dev_csv = pd.read_csv("./CSVfile/dev.csv", sep = "#")
## 分离文件路径、文本内容和标签

# def annotate_emotion_marks(text):
#     return re.sub(r"\[(.*?)\]", r"<<\1>>", text)
# def annotate_emotion_marks(text):
#     return re.sub(r"\[(.*?)\]", lambda match: f"[{match.group(1)}][{match.group(1)}][{match.group(1)}]", text)  
 
def preprocess_text_with_conversation_id(csv_file):
    """
    从CSV中读取数据，将对话名（id字段）加到文本内容前，并计算文本长度。
    :param csv_file: CSV文件路径
    :return: 新的文本列表和标签列表（含长度信息）
    """
    data = pd.read_csv(csv_file, sep="#")
    # data['text'] = data['text'].apply(annotate_emotion_marks)
    # 合并对话名和文本
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度
    
    # 添加文本长度到结果中
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    
    # 返回处理后的文本和标签
    return list(enhanced_with_length), list(data['label'])
    
# def preprocess_text_with_conversation_id(csv_file):
#     """
#     从CSV中读取数据，将对话名（id字段）加到文本内容前。
#     :param csv_file: CSV文件路径
#     :return: 新的文本列表和标签列表
#     """
#     data = pd.read_csv(csv_file, sep="#")
    
#     # 合并对话名和文本
#     data['enhanced_text'] = data['id'] + ": " + data['text']
    
#     # 返回处理后的文本和标签
#     return list(data['enhanced_text']), list(data['label'])


# 处理训练和验证集
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
# train_label = list(train_label)[:1500]
# train_txt = list(train_txt)[:1500]
# dev_label = list(dev_label)[:500]
# dev_txt = list(dev_txt)[:500]
# 文本向量化
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)

# 构建数据集
train_dataset = TensorDataset(train_coded_txt["input_ids"], 
                              train_coded_txt["attention_mask"],
                              torch.tensor(train_label))
dev_dataset = TensorDataset(dev_coded_txt["input_ids"], 
                            dev_coded_txt["attention_mask"],
                            torch.tensor(dev_label))
## 训练时间较长，建议可以先截取部分样本进行代码正确性验证，再使用全部样本
print(len(train_dataset),len(dev_dataset))

## 这里的batch_size 可以从1、2、4、8、16...尝试，过大的batch_size会使训练过程因为显存不足失败
batch_size = 16
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

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {0:"angry",1:"happy or excited",2:"neutral",3:"sad"}
    # print(preds)
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


def calculate_score_classification(preds, labels, average_f1='macro'):  # weighted, macro
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix


# class MyDLmodel():
#     def __init__(self, model, device, weight_decay=0.01, dropout_prob=0.3):
#         self.model = model

#         # 添加 Dropout
#         for module in self.model.modules():
#             if isinstance(module, nn.Linear):
#                 module.dropout = nn.Dropout(p=dropout_prob)

#         self.model.to(device)

#         # 添加权重惩罚 (L2 正则化)
#         self.optimizer = torch.optim.AdamW(
#             self.model.parameters(), 
#             lr=2e-5, 
#             eps=1e-8, 
#             weight_decay=weight_decay
#         )
        
#         self.scheduler = None
#         self.device = device

#     def evaluate(self, dataloader_val):
#         self.model.eval()
#         all_preds = []
#         all_labels = []
#         total_loss = 0

#         criterion = nn.CrossEntropyLoss()

#         for batch in dataloader_val:
#             input_ids = batch[0].to(self.device)
#             attention_mask = batch[1].to(self.device)
#             labels = batch[2].to(self.device)

#             with torch.no_grad():
#                 outputs = self.model(input_ids, attention_mask=attention_mask)
#                 loss = criterion(outputs.logits, labels)
#                 logits = outputs.logits

#             total_loss += loss.item()
#             all_preds.append(logits.detach().cpu().numpy())
#             all_labels.append(labels.detach().cpu().numpy())

#         avg_loss = total_loss / len(dataloader_val)
#         all_preds = np.concatenate(all_preds, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
        
#         accuracy, ua, f1, precision, confuse_matrix = calculate_score_classification(all_preds, all_labels)
#         return avg_loss, accuracy, ua, f1, precision, confuse_matrix

#     def train(self, dataloader_train, dataloader_dev, epochs):
#         criterion = nn.CrossEntropyLoss()
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0

#             for batch in dataloader_train:
#                 input_ids = batch[0].to(self.device)
#                 attention_mask = batch[1].to(self.device)
#                 labels = batch[2].to(self.device)

#                 self.optimizer.zero_grad()
#                 outputs = self.model(input_ids, attention_mask=attention_mask)
#                 loss = criterion(outputs.logits, labels)
#                 loss.backward()
#                 self.optimizer.step()

#                 total_loss += loss.item()

#             avg_train_loss = total_loss / len(dataloader_train)
#             print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

#             val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
#             print(f"Validation Loss: {val_loss:.4f}")
#             print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")

#     def predict(self, dataloader_test):
#         self.model.eval()
#         all_preds = []

#         for batch in dataloader_test:
#             input_ids = batch[0].to(self.device)
#             attention_mask = batch[1].to(self.device)

#             with torch.no_grad():
#                 outputs = self.model(input_ids, attention_mask=attention_mask)
#                 logits = outputs.logits

#             preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
#             all_preds.extend(preds)

#         return all_preds

#     def cross_validate(self, dataset, k_folds=5, epochs=3, batch_size=16):
#         kfold = KFold(n_splits=k_folds, shuffle=True)
#         results = []

#         for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
#             print(f"Fold {fold + 1}/{k_folds}")

#             train_subset = Subset(dataset, train_idx)
#             val_subset = Subset(dataset, val_idx)

#             dataloader_train = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#             dataloader_val = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

#             # 重新初始化模型和优化器（避免权重泄漏）
#             self.model.apply(self._reset_weights)

#             self.train(dataloader_train, dataloader_val, epochs)

#             val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_val)
#             print(f"Fold {fold + 1} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

#             results.append((val_loss, accuracy, ua, f1, precision))

#         return results

#     def _reset_weights(self, m):
#         if hasattr(m, 'reset_parameters'):
#             m.reset_parameters()

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
    

## 特征处理函数，可以对提取的特征进行处理，以获得更好的特征表示
def feature_process(feature):
    return feature

# 平衡损失函数
# class BalancedLoss(nn.Module):
#     def __init__(self, weights):
#         super(BalancedLoss, self).__init__()
#         self.weights = weights

#     def forward(self, logits, labels):
#         loss = F.cross_entropy(logits, labels, weight=self.weights)
#         return loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weights, reduction='none')
        p_t = torch.exp(-ce_loss)  # For each sample, p_t is the predicted probability for the true class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()  # Average over all batches

# 新增训练逻辑和增强功能
def train_with_augmentation(dataloader_train, dataloader_dev, model, device, epochs):
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # criterion = BalancedLoss(
    #     weights=torch.tensor(compute_class_weight('balanced', classes=np.unique(train_label), y=train_label), dtype=torch.float32).to(device)
    # )
    class_counts = np.array([606, 891, 1066, 696])
    N = class_counts.sum()

# 计算权重
    class_weights = N / class_counts
    print(class_weights)#双1.4比较好
    class_weights[0] *= 1.4  # 增加 'angry' 类别的权重
    class_weights[2] *= 1.4  # 增加 'neutral' 类别的权重
    class_weights[1] *= 1.0  # 不改变 'happy or excited'
    class_weights[3] *= 1.0  # 增加 'sad' 类别的权重（稍微增加）
        
# 转换为 tensor 并返回
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(
    alpha=1,  # 可以根据类别不平衡情况调整
    gamma=5,  # 可以通过实验调整
    weights=weights_tensor  # 使用你调整过的 class_weights
)   
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader_train):
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(dataloader_train):.4f}")
        _, accuracy, ua, f1, precision, confuse_matrix = evaluate(model, dataloader_train, device)
        print(f"Training Accuracy: {accuracy:.4f}, UA: {ua:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")
        writer.add_scalar('Loss/train', total_loss/len(dataloader_train), epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('F1/train', f1, epoch)
        writer.add_scalar('UA/train', ua, epoch)
        writer.add_scalar('Precision/train', precision, epoch)


        # val_loss, accuracy, f1 = evaluate(model, dataloader_dev, device)
        # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        val_loss, accuracy, ua, f1, precision, confuse_matrix = evaluate(model, dataloader_dev, device)

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        print(f"UA: {ua:.4f}, Precision: {precision:.4f}")
        print(f"Confusion Matrix:\n{confuse_matrix}")
        writer.add_scalar('Loss/dev', total_loss/len(dataloader_train), epoch)
        writer.add_scalar('Accuracy/dev', accuracy, epoch)
        writer.add_scalar('F1/dev', f1, epoch)
        writer.add_scalar('UA/dev', ua, epoch)
        writer.add_scalar('Precision/dev', precision, epoch)
def evaluate(model, dataloader_val, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader_val:
        # Move batch to device
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            logits = outputs.logits

        total_loss += loss.item()
        # Append predictions and true labels
        all_preds.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    # Average loss
    avg_loss = total_loss / len(dataloader_val)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate classification scores using the provided function
    accuracy, ua, f1, precision, confuse_matrix = calculate_score_classification(all_preds, all_labels)
    
    # Return results
    return avg_loss, accuracy, ua, f1, precision, confuse_matrix


# 主训练函数
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    epochs = 20
    set_seeds(17)
    # set_seeds(42)
    model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',
        num_labels=4,
        output_hidden_states=False
    ).to(device)
    total_steps = len(dataloader_train) * epochs


    train_with_augmentation(dataloader_train, dataloader_dev, model, device, epochs)

    writer.close()