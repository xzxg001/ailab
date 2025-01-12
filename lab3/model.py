import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as npW
import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
import torch.nn as nn
import random
import re
from transformers import BertForSequenceClassification, AdamW, get_cosine_schedule_with_warmup# 数据增强（过采样少数类）
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import RobertaForSequenceClassification,RobertaTokenizer, RobertaModel
from sklearn.utils.class_weight import compute_class_weight
import json
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


def text_tokenize(text_list): 
    # tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased',do_lower_case=True)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_text


def preprocess_text_with_conversation_id(csv_file):
    """
    从CSV中读取数据，将对话名（id字段）加到文本内容前，并计算文本长度。
    :param csv_file: CSV文件路径
    :return: 新的文本列表和标签列表（含长度信息）
    """
    data = pd.read_csv(csv_file, sep="#")
    # data['text'] = data['text'].apply(annotate_emotion_marks)
    # # 使用 VADER 对文本进行情感分析并生成情感标签
    # data['vader_label'] = data['text'].apply(apply_vader_sentiment)
    # 合并对话名和文本
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度
    # # 如果情感标签和原标签冲突，选择VADER的情感标签（根据需要）
    # data['final_label'] = data['vader_label']  # 使用VADER标签作为最终标签
    # 添加文本长度到结果中
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    
    # 返回处理后的文本和标签
    return list(enhanced_with_length), list(data['label'])

def preprocess_text_with_conversation_id_test(csv_file):
    """
    从CSV中读取数据，将对话名（id字段）加到文本内容前，并计算文本长度。
    :param csv_file: CSV文件路径
    :return: 新的文本列表和标签列表（含长度信息）
    """
    data = pd.read_csv(csv_file, sep="#")
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度

    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    
    # 返回处理后的文本和标签
    return list(enhanced_with_length)

# 处理训练和验证集
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
test_txt = preprocess_text_with_conversation_id_test("./CSVfile/test.csv")

# 文本向量化
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)
test_coded_txt = text_tokenize(test_txt)
# 构建数据集
train_dataset = TensorDataset(train_coded_txt["input_ids"], 
                              train_coded_txt["attention_mask"],
                              torch.tensor(train_label))
dev_dataset = TensorDataset(dev_coded_txt["input_ids"], 
                            dev_coded_txt["attention_mask"],
                            torch.tensor(dev_label))
test_dataset = TensorDataset(test_coded_txt["input_ids"], 
                             test_coded_txt["attention_mask"])


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
dataloader_test = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=32
)

# 定义模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
model.to(device)
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
# In MyDLmodel class
    def __init__(self, model, device, weight_decay=0.01, num_training_steps=None, num_warmup_steps=0, lr=2e-5,alpha=1, gamma=2, params=None):
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay,amsgrad=True)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
        class_weights[0] *= 1.4  # 增加 'angry' 类别的权重
        class_weights[2] *= 1.4  # 增加 'neutral' 类别的权重
        class_weights[1] *= 1.0  # 不改变 'happy or excited'
        class_weights[3] *= 1.0  # 增加 'sad' 类别的权重（稍微增加）
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, weights=weights_tensor)
        self.device = device
        self.params = params  # 当前超参数
        self.best_f1 = 0.0  # 历史最佳 F1 分数
        self.best_params = None  # 历史最佳超参数

# In training loop
    def train(self, dataloader_train, dataloader_dev, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader_train:
                input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
        # Rest of the training loop

            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f},confuse_matrix:\n{confuse_matrix}")
                        # 如果当前 F1 分数更高，则更新历史最佳 F1 分数和超参数
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_params = self.params  # 更新历史最佳超参数
                # 保存最佳超参数到文件
                with open('best_params.json', 'w') as f:
                    json.dump(self.best_params, f)
                print(f"New best F1 achieved: {f1:.4f}. Best parameters saved.") 

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
    test_csv.to_csv("./CSVfile/result.csv", sep="#")
    print("测试集预测结果已成功写入到文件中！")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seeds(17)
    pretrained_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, output_hidden_states=True).to(device)

    # 计算总训练步数
    epochs = 20
    total_steps = len(dataloader_train) * 20 # dataloader_train 必须提前定义
    all_warmup_steps = int(total_steps * 0.12852249060729312)
    # mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps,num_warmup_steps=all_warmup_steps,
    # lr=2e-5, alpha=1, gamma=2
    # )
    mymodel = MyDLmodel(
    pretrained_model, 
    device, 
    weight_decay=0.005406670341704483, 
    num_training_steps=total_steps, 
    num_warmup_steps=all_warmup_steps, 
    lr=7.182477201000406e-06, 
    alpha=1.4306490712836057, 
    gamma=1.5952636368058761
)
    from torchviz import make_dot

# 示例输入
    input_ids = torch.randint(0, 100, (1, 256)).to(device)  # 模拟输入
    attention_mask = torch.ones((1, 256)).to(device)  # 模拟注意力掩码

# 获取模型输出
    outputs = mymodel.model(input_ids, attention_mask=attention_mask)

# 生成计算图
    dot = make_dot(outputs.logits, params=dict(mymodel.model.named_parameters()))

# 保存为图片
    dot.format = 'png'
    dot.render('model_architecture')
