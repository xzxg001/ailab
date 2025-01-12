import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
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
from transformers import BertForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
import json
from torchview import draw_graph
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import MultiheadAttention
from hyperopt import fmin, tpe, hp, Trials, space_eval
from funasr import AutoModel

# 设置随机种子
def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# 文本分词
def text_tokenize(text_list):
    tokenizer = AutoTokenizer.from_pretrained("meghanadh/finetune_bert_iemocap_text")
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_text

# 数据预处理
def preprocess_text_with_conversation_id(csv_file):
    data = pd.read_csv(csv_file, sep="#")
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    return list(enhanced_with_length), list(data['label'])

def preprocess_text_with_conversation_id_test(csv_file):
    data = pd.read_csv(csv_file, sep="#")
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    return list(enhanced_with_length)

# 提取音频特征
def extract_audio_feature(file_list, save_path, granularity="utterance"):
    if os.path.exists(save_path):
        print(f"已找到保存的特征文件 '{save_path}'，正在加载...")
        feature = np.load(save_path)
        print("特征加载完毕！")
        return feature

    print("请耐心等待特征提取完！")
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(model=model_id, hub="ms", disable_update=True)
    feature = []
    for n, file in enumerate(file_list):
        rec_result = model.generate(file, output_dir="./outputs", granularity=granularity)
        if isinstance(rec_result, list) and len(rec_result) > 0:
            feats = rec_result[0].get('feats', None)
        else:
            feats = None
        if feats is None:
            raise ValueError(f"无法从文件 {file} 中提取特征，返回值格式不正确: {rec_result}")
        if granularity == "utterance":
            feats = np.array(feats).reshape(-1)
        elif granularity == "frame":
            feats = np.array(feats)
        feature.append(feats)
        if (n + 1) % 100 == 0:
            print(f"当前进度 {n + 1}/{len(file_list)}")
    print("此次特征提取已结束")
    print("-------------------------------")
    if granularity == "utterance":
        feature = np.stack(feature, axis=0)
    elif granularity == "frame":
        feature = np.array(feature)
    np.save(save_path, feature)
    print(f"特征已保存到文件 '{save_path}'")
    return feature
train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/ans.csv")
dev1_txt, dev1_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
test_txt = preprocess_text_with_conversation_id_test("./CSVfile/test.csv")
train_csv = pd.read_csv("./CSVfile/train.csv", sep = "#")
dev_csv = pd.read_csv("./CSVfile/test.csv", sep = "#")
dev1_csv = pd.read_csv("./CSVfile/dev.csv", sep = "#")
test_csv = pd.read_csv("./CSVfile/test.csv", sep = "#")
train_path = list(train_csv.path)
train_label = list(train_label)
train_txt = list(train_txt)

dev_path = list(dev_csv.path)
dev_label = list(dev_label)
dev_txt = list(dev_txt)

dev1_path = list(dev1_csv.path)
dev1_label = list(dev1_label)
dev1_txt = list(dev1_txt)

test_path = list(test_csv.path)
test_txt = list(test_txt)

train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)
dev1_coded_txt = text_tokenize(dev1_txt)
test_coded_txt = text_tokenize(test_txt)

train_audio_features = extract_audio_feature(train_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/train_feature.npy")
train_dataset = TensorDataset(
    train_coded_txt["input_ids"], 
    train_coded_txt["attention_mask"], 
    torch.tensor(train_audio_features), 
    torch.tensor(train_label)
)

# 验证数据集
dev_audio_features = extract_audio_feature(dev_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/test_feature.npy")
dev_dataset = TensorDataset(
    dev_coded_txt["input_ids"], 
    dev_coded_txt["attention_mask"], 
    torch.tensor(dev_audio_features), 
    torch.tensor(dev_label)
)
dev1_audio_features = extract_audio_feature(dev1_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/dev_feature.npy")
dev1_dataset = TensorDataset(
    dev1_coded_txt["input_ids"], 
    dev1_coded_txt["attention_mask"], 
    torch.tensor(dev1_audio_features), 
    torch.tensor(dev1_label)
)

test_audio_features = extract_audio_feature(test_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/test_feature.npy")
test_dataset = TensorDataset(test_coded_txt["input_ids"], 
                             test_coded_txt["attention_mask"],    
                             torch.tensor(test_audio_features) )
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
dataloader_dev1 = DataLoader(
    dev1_dataset,
    sampler=RandomSampler(dev1_dataset),
    batch_size=32
)
dataloader_test = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=32
)
# 定义模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# 定义 Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weights, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# 定义组合模型
class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, num_labels=4):
        super(CombinedModel, self).__init__()
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.text_projection = nn.Linear(768, 1024)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256 * 512, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, audio_features):
        roberta_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = roberta_output.hidden_states[-1]
        text_features = last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        combined_features = torch.cat((text_features, audio_features), dim=1)
        combined_features = combined_features.unsqueeze(1)
        x = F.relu(self.conv1(combined_features))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# 定义训练和评估类
class MyDLmodel:
    def __init__(self, model, device, params, weight_decay=0.01, num_training_steps=None, num_warmup_steps=0):
        self.model = model
        self.model.to(device)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if 'pretrained_model' not in n], 'weight_decay': params['weight_decay']}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=params['alpha'], gamma=params['gamma'], weights=weights_tensor)
        self.device = device
        self.params = params
        self.best_f1 = 0.0
        self.best_params = None

    def train(self, dataloader_train, dataloader_dev, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader_train:
                input_ids, attention_mask, audio_features, labels = tuple(t.to(device) for t in batch)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, audio_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(dataloader_train)
            # writer.add_scalar('Train/Loss', avg_train_loss, epoch+1)
            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Confusion Matrix:\n{confuse_matrix}")
            val1_loss, val1_accuracy, val1_ua, val1_f1, val1_precision, val1_confuse_matrix = self.evaluate(dataloader_dev1)
            print(f"Epoch {epoch+1}: Validation1 Loss: {val1_loss:.4f}, Accuracy: {val1_accuracy:.4f}, F1 Score: {val1_f1:.4f}, Precision: {val1_precision:.4f}, Confusion Matrix:\n{val1_confuse_matrix}")
            # writer.add_scalar('Validation/Loss', val_loss, epoch+1)
            # writer.add_scalar('Validation/Accuracy', accuracy, epoch+1)
            # writer.add_scalar('Validation/F1_Score', f1, epoch+1)
            # writer.add_scalar('Validation/Precision', precision, epoch+1)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_params = self.params
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
            input_ids, attention_mask, audio_features, labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, audio_features)
                loss = criterion(outputs, labels)
                logits = outputs
            total_loss += loss.item()
            all_preds.append(logits.detach().cpu().numpy())
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
            input_ids, attention_mask, audio_features = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, audio_features)
                logits = outputs
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            all_preds.extend(preds)
        return all_preds

# 定义目标函数
def objective(params):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text", num_labels=4, output_hidden_states=True)
    model = CombinedModel(pretrained_model, num_labels=4).to(device)
    epochs = 10
    total_steps = len(dataloader_train) * epochs
    all_warmup_steps = int(total_steps * 0.1)
    mymodel = MyDLmodel(
        model, 
        device, 
        params=params, 
        num_training_steps=total_steps, 
        num_warmup_steps=all_warmup_steps
    )
    mymodel.train(dataloader_train, dataloader_dev, epochs)
    val_loss, accuracy, ua, f1, precision, confuse_matrix = mymodel.evaluate(dataloader_dev)
    return 1 - f1

# 定义搜索空间
space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(5e-5)),
    'weight_decay': hp.uniform('weight_decay', 0.0, 0.1),
    'alpha': hp.uniform('alpha', 0.5, 2.0),
    'gamma': hp.uniform('gamma', 1.0, 3.0)
}

# 主函数
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seeds(17)
    #Best Hyperparameters: {'alpha': 1.9804152891237692, 'gamma': 1.4789610739401222, 'lr': 1.8991294150802777e-05, 'weight_decay': 0.020968676503784745}
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    print("Best Hyperparameters:", best)
    best_params = space_eval(space, best)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text", num_labels=4, output_hidden_states=True)
    model = CombinedModel(pretrained_model, num_labels=4).to(device)
    epochs = 32
    total_steps = len(dataloader_train) * epochs
    all_warmup_steps = int(total_steps * 0.1)
    mymodel = MyDLmodel(
        model, 
        device, 
        params=best_params, 
        num_training_steps=total_steps, 
        num_warmup_steps=all_warmup_steps
    )
    writer = SummaryWriter('/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/lab4/log/combined_model')
    mymodel.train(dataloader_train, dataloader_dev, epochs)
    writer.close()
    test_preds = mymodel.predict(dataloader_test)
    write_result(test_preds)
    test_df = pd.read_csv("./CSVfile/result.csv", sep="#")
    test_labels = test_df['label'].tolist()
    ans_df = pd.read_csv("./CSVfile/ans.csv", sep="#")
    true_labels = ans_df['label'].tolist()
    true_labels = [int(label) for label in true_labels]
    f1 = f1_score(true_labels, test_labels, average='macro')
    print(f"测试集上的宏平均F1分数为: {f1:.4f}")