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
from pathlib import Path
from transformers import RobertaForSequenceClassification,RobertaTokenizer, RobertaModel
from sklearn.utils.class_weight import compute_class_weight
import json
from torch.utils.tensorboard import SummaryWriter
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


# 处理训练和验证集
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
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

def load_checkpoint(checkpoint_path):
    best_f1 = 0
    val_loss = float('inf')
    accuracy = 0
    ua = 0
    precision = 0
    cm_array = np.zeros((4, 4), dtype=int)  # 将 confusion_matrix 改为 cm_array
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)  # 加载检查点
            best_f1 = checkpoint.get('best_f1', 0)
            val_loss = checkpoint.get('val_loss', float('inf'))
            accuracy = checkpoint.get('accuracy', 0)
            ua = checkpoint.get('ua', 0)
            precision = checkpoint.get('precision', 0)
            cm_array = np.array(checkpoint.get('confusion_matrix', np.zeros((4, 4), dtype=int)))
        except Exception as e:
            print(f"加载检查点时出错: {e}")
    return best_f1, val_loss, accuracy, ua, precision, cm_array  # 返回 cm_array

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, output_dim, dropout=0.5):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (N, 1, seq_len, embedding_dim)
        conved = [conv(x) for conv in self.convs]  # (N, C, H, 1)
        conved = [conv.squeeze(3) for conv in conved]  # (N, C, H)
        conved = [self.bns[i](conv) for i, conv in enumerate(conved)]  # Apply BN
        conved = [F.relu(conv) for conv in conved]  # Apply ReLU
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # (N, C)
        pooled = [self.dropout(p) for p in pooled]  # Apply dropout
        cat = torch.cat(pooled, dim=1)  # (N, C * len(filter_sizes))
        return self.fc(cat)

    
class MyDLmodel:
    def __init__(self, model, device, weight_decay=0.01, dropout_prob=0.1,num_training_steps=None, num_warmup_steps=0, lr=2e-5,alpha=1, gamma=2, params=None,best_f1=0):
        self.model = model
        self.best_f1 = best_f1  # Initialize best_f1
        print(f"Current best_f1: {self.best_f1:.4f}")

        self.textcnn = TextCNN(
            embedding_dim=768,  # BERT's hidden size
            num_filters=100,  # Number of filters per filter size
            filter_sizes=[2, 3, 4],  # Sizes of filters
            output_dim=4,  # Number of classes
            dropout=dropout_prob
        ).to(device)

        self.model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.textcnn.parameters()), lr=lr, weight_decay=weight_decay)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        class_counts = np.array([606, 891, 1066, 696])
        N = class_counts.sum()
        class_weights = N / class_counts
        class_weights[0] *= 1.4
        class_weights[2] *= 1.4
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, weights=weights_tensor)
        # self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    def train(self, dataloader_train, dataloader_dev, epochs):
        best_f1 = 0
        for epoch in range(epochs):
            self.model.train()
            self.textcnn.train()
            total_loss = 0

            for step, batch in enumerate(dataloader_train):
                input_ids, attention_mask, labels = tuple(t.to(self.device) for t in batch)

                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
                bert_features = outputs.hidden_states[-1]

                cnn_output = self.textcnn(bert_features)
                loss = self.criterion(cnn_output, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

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
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.val_loss = val_loss
                self.accuracy = accuracy
                self.ua = ua
                self.precision = precision
                self.confusion_matrix = confuse_matrix
                torch.save({
    'bert_model_state_dict': self.model.state_dict(),
    'textcnn_state_dict': self.textcnn.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'best_f1': self.best_f1,
    'val_loss': self.val_loss,
    'accuracy': self.accuracy,
    'ua': self.ua,
    'precision': self.precision,
    'confusion_matrix': self.confusion_matrix.tolist()  # 确保混淆矩阵以列表形式保存
                }, model_dir / f"model_best.pt")

    def evaluate(self, dataloader_val):
        self.model.eval()
        self.textcnn.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in dataloader_val:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
                bert_features = outputs.hidden_states[-1]
                cnn_output = self.textcnn(bert_features)
                loss = self.criterion(cnn_output, labels)

            total_loss += loss.item()
            all_preds.append(cnn_output.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader_val)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

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


if __name__  == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    set_seeds(17)
    # 模型存储路径
    model_dir = Path("./model/bert_checkpoints")
# 如果模型目录不存在，则创建一个
    os.makedirs(model_dir) if not os.path.exists(model_dir) else ''
    checkpoint_path = model_dir / "model_best.pt"
    print(f"Checkpoint path: {checkpoint_path}") 
    # pretrained_model = BertForSequenceClassification.from_pretrained(
    #     './bert-base-uncased',
    #     num_labels=4,
    #     output_hidden_states=True
    #     ).to(device)
    pretrained_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, output_hidden_states=True).to(device)
    writer = SummaryWriter('/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/lab3/log/bert+textcnn')
    # 计算总训练步数
    epochs = 20
    total_steps = len(dataloader_train) * epochs  # dataloader_train 必须提前定义
    all_warmup_steps = int(total_steps * 0.1)
    best_f1, val_loss, accuracy, ua, precision, cm = load_checkpoint(checkpoint_path)
    # mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps,best_f1=best_f1)
    mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps,num_warmup_steps=all_warmup_steps,
    lr=2e-5, alpha=1, gamma=2
    )
    
    mymodel.train(dataloader_train, dataloader_dev, epochs)
    writer.close()
