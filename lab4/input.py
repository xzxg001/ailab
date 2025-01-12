import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch import nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
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
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    return encoded_text
## 'input_ids' 'token_type_ids' 'attention_mask'
## 提取特征函数
## file_list:音频文件路径的列表  list类型
## 返回值numpy.ndarray  形状:(len(file_list),88)
def extract_audio_feature(file_list,save_path):
    # 如果已存在保存的特征文件，则直接加载
    if os.path.exists(save_path):
        print(f"已找到保存的特征文件 '{save_path}'，正在加载...")
        feature = np.load(save_path)
        print("特征加载完毕！")
        return feature    
    print("请耐心等待特征提取完！")
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals)#指定了特征集（eGeMAPSv02）和特征级别（Functionals）
    feature = []
    for n,file in enumerate(file_list):#enumerate函数用于遍历，返回一个迭代器（索引和值）
        y = smile.process_file(file)#调用process_file函数处理音频文件
        y = y.to_numpy().reshape(-1)#-1 表示自动计算这一维的大小
        feature.append(y)
        if (n+1)%100 == 0:
            print(f"当前进度{n+1}/{len(file_list)}")
    print("此次特征提取已结束")
    print("-------------------------------")
    feature = np.stack(feature,axis = 0)

    # 将特征保存到指定的文件中
    np.save(save_path, feature)
    print(f"特征已保存到文件 '{save_path}'")
    
    return feature
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
# 添加新的特殊token
special_tokens_dict = {'additional_special_tokens': ['[AUDIO]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'special tokens')
def preprocess_text_with_conversation_id(csv_file):
    """
    从CSV中读取数据，将对话名（id字段）加到文本内容前，并计算文本长度。
    :param csv_file: CSV文件路径
    :return: 新的文本列表和标签列表（含长度信息）
    """
    data = pd.read_csv(csv_file, sep="#")
    # 合并对话名和文本，并添加 [AUDIO] token
    data['enhanced_text'] = '[AUDIO] ' + data['id'].astype(str) + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度
    # 添加文本长度到结果中
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    
    # 返回处理后的文本和标签
    return list(enhanced_with_length), list(data['label'])
    
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
train_csv = pd.read_csv("./CSVfile/train.csv", sep = "#")
dev_csv = pd.read_csv("./CSVfile/dev.csv", sep = "#")
train_path = list(train_csv.path)
train_label = list(train_label)
train_txt = list(train_txt)
dev_path = list(dev_csv.path)
dev_label = list(dev_label)
dev_txt = list(dev_txt)
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)

train_audio_features = extract_audio_feature(train_path, "train_audio_features.npy")
train_dataset = TensorDataset(
    train_coded_txt["input_ids"], 
    train_coded_txt["attention_mask"], 
    torch.tensor(train_audio_features), 
    torch.tensor(train_label)
)

# 验证数据集
dev_audio_features = extract_audio_feature(dev_path, "dev_audio_features.npy")
dev_dataset = TensorDataset(
    dev_coded_txt["input_ids"], 
    dev_coded_txt["attention_mask"], 
    torch.tensor(dev_audio_features), 
    torch.tensor(dev_label)
)

## 这里的batch_size 可以从1、2、4、8、16...尝试，过大的batch_size会使训练过程因为显存不足失败
batch_size = 4
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
class MultiModalModel(nn.Module):
    def __init__(self, text_model, audio_feature_dim, num_classes):
        super(MultiModalModel, self).__init__()
        self.text_model = text_model
        # 扩展BERT模型的词表
        self.text_model.resize_token_embeddings(len(tokenizer))
        # 添加音频特征映射层
        self.audio_fc = nn.Linear(audio_feature_dim, text_model.config.hidden_size)
        self.classifier = nn.Linear(text_model.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, audio_features):
        # 获取BERT的token embeddings
        input_embeds = self.text_model.bert.embeddings(input_ids)
        # 假设 [AUDIO] token是每个序列的第一个token
        audio_embed = self.audio_fc(audio_features)
        # 替换 [AUDIO] token的嵌入
        input_embeds[:, 0, :] = audio_embed
        # 前向传播通过BERT模型
        outputs = self.text_model.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
class MyDLmodel():
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                          num_warmup_steps=0, 
                                                          num_training_steps=len(dataloader_train)*epochs)
        self.device = device

    def train(self, dataloader_train, dataloader_dev, epochs):
        total_steps = len(dataloader_train) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                     num_warmup_steps=0, 
                                                     num_training_steps=total_steps)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}"):
                input_ids, attention_mask, audio_features, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                audio_features = audio_features.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, audio_features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
            self.evaluate(dataloader_dev)

    def evaluate(self, dataloader_dev):
        self.model.eval()
        eval_loss = 0
        predictions, true_labels = [], []
        for batch in tqdm(dataloader_dev, desc="Evaluating"):
            input_ids, attention_mask, audio_features, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            audio_features = audio_features.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, audio_features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                eval_loss += loss.item()
                logits = outputs.detach().cpu().numpy()
                label_ids = labels.detach().cpu().numpy()
                predictions.append(logits)
                true_labels.append(label_ids)
        avg_eval_loss = eval_loss / len(dataloader_dev)
        print(f"Average evaluation loss: {avg_eval_loss:.4f}")
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        accuracy, ua, f1, precision, confuse_matrix = calculate_score_classification(predictions, true_labels)
        print(f"Accuracy: {accuracy:.4f}, UA: {ua:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")
        print("Confusion Matrix:")
        print(confuse_matrix)
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

def feature_process(feature):
    return feature
if __name__ == "__main__":
    set_seeds(17)
    bert_model = BertForSequenceClassification.from_pretrained('./bert-base-uncased')
    audio_feature_dim = 88  # 根据提取的音频特征维度调整
    num_classes = 4
    epochs = 20
    model = MultiModalModel(bert_model, audio_feature_dim, num_classes)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    mymodel = MyDLmodel(model, device)
    
    mymodel.train(dataloader_train, dataloader_dev, epochs)