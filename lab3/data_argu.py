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
import re
from transformers import BertForSequenceClassification, AdamW, get_cosine_schedule_with_warmup# 数据增强（过采样少数类）
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from nltk.sentiment.vader import SentimentIntensityAnalyzer
## 文本向量化函数
## text_list 文本内容的list
## 返回一个字典，{'input_ids':value, 'token_type_ids':value, 'attention_mask':value},每个元素的长度等于len(text_list)
## "input_ids"-词转换为数字后的序列 'token_type_ids'-标记一段文本中不同句子的序号 'attention_mask'-标记填充位置的序号 
import pandas as pd
from transformers import BertTokenizer
from nlpaug.augmenter.word import SynonymAug, RandomWordAug
from nlpaug.flow import Sometimes

# 文本分词函数
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

# 数据增强函数
def augment_text(text_list, aug_prob=0.3):
    """
    对文本列表进行数据增强
    :param text_list: 文本列表
    :param aug_prob: 每条文本被增强的概率
    :return: 增强后的文本列表
    """
    synonym_aug = SynonymAug(aug_src='wordnet', aug_p=aug_prob)
    random_aug = RandomWordAug(action="swap", aug_p=aug_prob)
    augmenter = Sometimes([synonym_aug, random_aug])  # 随机使用一种增强器
    
    augmented_texts = []
    for text in text_list:
        augmented_text = augmenter.augment(text)
        if isinstance(augmented_text, list):  # 增强输出为列表时，取第一个
            augmented_text = augmented_text[0]
        augmented_texts.append(augmented_text)
    
    return augmented_texts

# 修改预处理函数，加入数据增强
def preprocess_text_with_conversation_id(csv_file, aug_prob=0.3):
    """
    从CSV中读取数据，将对话名（id字段）加到文本内容前，并计算文本长度。
    :param csv_file: CSV文件路径
    :param aug_prob: 数据增强的概率
    :return: 新的文本列表和标签列表（含长度信息）
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file, sep="#")
    
    # 合并对话名和文本
    data['enhanced_text'] = data['id'].astype(str) + ": " + data['text']
    data['text_length'] = data['enhanced_text'].apply(len)  # 计算合并后文本的长度
    data['enhanced_text_with_length'] = data['enhanced_text'] + " (长度：" + data['text_length'].astype(str) + ")"
    
    # 获取需要增强的文本列表
    texts_to_augment = list(data['enhanced_text_with_length'])
    
    # 进行数据增强
    augmented_texts = augment_text(texts_to_augment, aug_prob=aug_prob)
    
    # 返回增强后的文本和标签
    return augmented_texts, list(data['label'])


# 处理训练和验证集
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv",aug_prob=0.3)
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv",aug_prob=0.3)

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
    label_dict_inverse = {0:"angry", 1:"happy or excited", 2:"neutral", 3:"sad"}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def calculate_score_classification(preds, labels, average_f1='macro'):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix

# class BalancedLoss(torch.nn.Module):
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

class MyDLmodel:
    def __init__(self, model, device, weight_decay=0.01, dropout_prob=1, num_training_steps=None):
        self.model = model
        # 添加 Dropout
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.dropout = nn.Dropout(p=dropout_prob)
        self.model.to(device)

        # 默认优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    #     self.criterion = BalancedLoss(
    #     weights=torch.tensor(compute_class_weight('balanced', classes=np.unique(train_label), y=train_label), dtype=torch.float32).to(device)
    # ) 
    # 样本数
        class_counts = np.array([606, 891, 1066, 696])
        N = class_counts.sum()

# 计算权重
        class_weights = N / class_counts
        print(class_weights)#1.4 1.4 1.0 1.0 
        class_weights[0] *= 1.4  # 增加 'angry' 类别的权重
        class_weights[2] *= 1.4  # 增加 'neutral' 类别的权重
        class_weights[1] *= 1.0  # 不改变 'happy or excited'
        class_weights[3] *= 1.0  # 增加 'sad' 类别的权重（稍微增加）
        
# 转换为 tensor 并返回
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(
    alpha=1,  # 可以根据类别不平衡情况调整
    gamma=5,  # 可以通过实验调整
    weights=weights_tensor  # 使用你调整过的 class_weights
)   
        # self.criterion = nn.CrossEntropyLoss()   
        self.device = device

    def train(self, dataloader_train, dataloader_dev, epochs, use_augmentation=False, train_labels=None):

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(dataloader_train):
                input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f},confuse_matrix:\n{confuse_matrix}")
            

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
        accuracy_per_class(all_preds, all_labels)
        accuracy, ua, f1, precision, confuse_matrix = calculate_score_classification(all_preds, all_labels)
        return avg_loss, accuracy, ua, f1, precision, confuse_matrix

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


    # def _reset_weights(self, m):
    #     if hasattr(m, 'reset_parameters'):
    #         m.reset_parameters()

def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误！请检查test_preds长度是否为1241！！！")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./result.csv", sep="#")
    print("测试集预测结果已成功写入到文件中！")

# 主训练函数

## model reference： https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForSequenceClassification
if __name__  == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    set_seeds(17)
    pretrained_model = BertForSequenceClassification.from_pretrained(
        './bert-base-uncased',
        num_labels=4,
        output_hidden_states=True
        ).to(device)
    
    # 计算总训练步数
    epochs = 20
    total_steps = len(dataloader_train) * epochs  # dataloader_train 必须提前定义
    mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps)

    mymodel.train(dataloader_train, dataloader_dev, epochs)



