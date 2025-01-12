import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import opensmile
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AdamW

# 1. 定义多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, bert_model, audio_feature_size, num_labels):
        super(MultimodalModel, self).__init__()
        
        # BERT部分
        self.bert = bert_model
        
        # 音频特征处理
        self.audio_fc = nn.Linear(audio_feature_size, 128)
        
        # 特征融合
        self.fc1 = nn.Linear(768 + 128, 256)
        self.fc2 = nn.Linear(256, num_labels)
    
    def forward(self, input_ids, attention_mask, audio_features):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls_output = bert_output.last_hidden_state[:, 0, :]
        
        # 音频特征处理
        audio_output = self.audio_fc(audio_features)
        
        # 拼接BERT和音频特征
        combined_output = torch.cat((bert_cls_output, audio_output), dim=1)
        
        # 分类层
        x = torch.relu(self.fc1(combined_output))
        x = self.fc2(x)
        
        return x

# 2. 音频特征提取函数
def extract_audio_feature(file_list, save_path):
    if os.path.exists(save_path):
        print(f"Found saved feature file '{save_path}', loading it...")
        feature = np.load(save_path)
        print("Features loaded!")
        return feature
    print("Extracting audio features, please wait...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    feature = []
    for n, file in enumerate(file_list):
        y = smile.process_file(file)
        y = y.to_numpy().reshape(-1)
        feature.append(y)
        if (n + 1) % 100 == 0:
            print(f"Progress: {n + 1}/{len(file_list)}")
    feature = np.stack(feature, axis=0)
    np.save(save_path, feature)
    print(f"Features saved to '{save_path}'")
    return feature

# 3. 文本特征处理函数
def text_tokenize(text_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors='pt'
    )
    return encoded_text

# 4. 创建数据加载器
def create_dataloader(texts, audio_files, labels, batch_size=4, save_path=None):
    # 提取音频特征
    audio_features = extract_audio_feature(audio_files, save_path=save_path)
    audio_features = torch.tensor(audio_features, dtype=torch.float32)
    
    # 文本特征处理
    text_features = text_tokenize(texts)
    
    # 确保所有张量的第一个维度一致
    assert audio_features.shape[0] == text_features['input_ids'].shape[0] == len(labels), "Mismatch in number of samples"
    
    dataset = TensorDataset(
        text_features['input_ids'], 
        text_features['attention_mask'], 
        audio_features, 
        torch.tensor(labels, dtype=torch.long)
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 5. 训练函数
def train_model(model, train_dataloader, val_dataloader, device, epochs=10):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-6, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, audio_features, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, audio_features=audio_features)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
        
        # 验证集评估
        evaluate_model(model, val_dataloader, device)

# 6. 评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, audio_features, label = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, audio_features=audio_features)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# 7. 主函数
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载BERT模型
    pretrained_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    
    # 初始化多模态模型
    model = MultimodalModel(bert_model=pretrained_model, audio_feature_size=88, num_labels=4)
    
    # 读取数据
    train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
    train_texts = list(train_csv['text'])[:1500]
    train_audio_files = list(train_csv['path'])[:1500]
    train_labels = list(train_csv['label'])[:1500]
    
    val_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")
    val_texts = list(val_csv['text'])[:500]
    val_audio_files = list(val_csv['path'])[:500]
    val_labels = list(val_csv['label'])[:500]
    
    # 确保音频文件存在
    for file in train_audio_files + val_audio_files:
        if not os.path.exists(file):
            print(f"Audio file not found: {file}")
            exit(1)
    
    # 创建数据加载器，使用不同的save_path
    train_dataloader = create_dataloader(train_texts, train_audio_files, train_labels, batch_size=4, save_path="./audio_features_train.npy")
    val_dataloader = create_dataloader(val_texts, val_audio_files, val_labels, batch_size=4, save_path="./audio_features_val.npy")
    
    # 训练模型
    train_model(model, train_dataloader, val_dataloader, device, epochs=10)