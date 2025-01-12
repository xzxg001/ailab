import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

# 提取特征函数（保持不变）
def extract_audio_feature(file_list, save_path):
    if os.path.exists(save_path):
        print(f"已找到保存的特征文件 '{save_path}'，正在加载...")
        feature = np.load(save_path)
        print("特征加载完毕！")
        return feature    
    print("请耐心等待特征提取完！")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals)
    feature = []
    for n, file in enumerate(file_list):
        y = smile.process_file(file)
        y = y.to_numpy().reshape(-1)
        feature.append(y)
        if (n + 1) % 100 == 0:
            print(f"当前进度{n + 1}/{len(file_list)}")
    print("此次特征提取已结束")
    feature = np.stack(feature, axis=0)
    np.save(save_path, feature)
    print(f"特征已保存到文件 '{save_path}'")
    return feature

# 归一化特征（保持不变）
def normalize_features(features, scaler=None, method='minmax'):
    if method == 'standard':
        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        print("标准化（Z-score）完成")
    
    elif method == 'minmax':
        if scaler is None:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        print("最小-最大归一化完成")
    
    else:
        raise ValueError("未知的归一化方法，请选择 'standard' 或 'minmax'。")
    
    return features, scaler

# 计算分类评估指标（保持不变）
def calculate_score_classification(preds, labels, average_f1='macro'):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix

class RNNEmotionClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_classes=4, num_layers=2, dropout=0.5):
        super(RNNEmotionClassifier, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 检查输入维度，如果是2D (batch_size, input_dim)，则添加一个维度来变成 (batch_size, seq_len, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 增加一个维度，变成 (batch_size, 1, input_dim)

        # 输入形状为(batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        
        # 使用LSTM的最后一个时刻的输出
        out = lstm_out[:, -1, :]  # 获取最后一个时刻的输出
        out = self.dropout(out)  # Dropout
        out = self.fc(out)  # 分类输出层
        return out


class RNNModel:
    def __init__(self, input_dim=88, num_classes=4, hidden_dim=128, num_layers=2, learning_rate=0.001, dropout=0.5):
        self.model = RNNEmotionClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, train_features, train_labels, epochs=10, batch_size=32):
        train_tensor = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    def evaluate(self, features, labels):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(features, dtype=torch.float32))
            _, preds = torch.max(outputs, 1)
        
        accuracy = accuracy_score(labels, preds.numpy())
        f1 = f1_score(labels, preds.numpy(), average='macro')
        precision = precision_score(labels, preds.numpy(), average='macro')
        recall = recall_score(labels, preds.numpy(), average='macro')
        conf_matrix = confusion_matrix(labels, preds.numpy())
        
        return accuracy, recall, f1, precision, conf_matrix

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(features, dtype=torch.float32))
            _, preds = torch.max(outputs, 1)
        return preds.numpy()

# 数据加载器创建函数
def create_dataloader(features, labels, batch_size=32):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 主函数
if __name__ == "__main__":
    # 读取数据
    train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
    dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")

    train_path = list(train_csv.path)
    train_label = list(train_csv.label)
    dev_path = list(dev_csv.path)
    dev_label = list(dev_csv.label)
    
    # 特征提取与归一化
    train_save_path = "./feature/train_feature.npy"
    train_feature = extract_audio_feature(train_path, train_save_path)
    train_feature_normalized, scaler = normalize_features(train_feature, method='minmax')
    
    # 创建训练数据加载器
    train_loader = create_dataloader(train_feature_normalized, np.array(train_label))
    
    # 模型定义与训练
    model = RNNModel(input_dim=train_feature_normalized.shape[1], num_classes=4)
    model.train(train_feature_normalized, np.array(train_label), epochs=20, batch_size=32)
    
    # 训练集评估
    train_preds = model.predict(train_feature_normalized)
    acc, ua, f1, pre, confuse_matrix = calculate_score_classification(train_preds, np.array(train_label))
    print(f"train:\nAcc:{acc} \nUa:{ua} \nMacro_F1:{f1} \nPre:{pre}\nConfuse_matrix:\n{confuse_matrix}")

    # 开发集评估
    dev_save_path = "./feature/dev_feature.npy"
    dev_feature = extract_audio_feature(dev_path, dev_save_path)
    dev_feature_normalized, _ = normalize_features(dev_feature, scaler=scaler, method='minmax')
    
    dev_preds = model.predict(dev_feature_normalized)
    acc, ua, f1, pre, confuse_matrix = calculate_score_classification(dev_preds, np.array(dev_label))
    print(f"dev:\nAcc:{acc} \nUa:{ua} \nMacro_F1:{f1} \nPre:{pre}\nConfuse_matrix:\n{confuse_matrix}")
    
    # 测试集预测部分
    # test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    # test_path = list(test_csv.path)
    # test_save_path = "./feature/test_feature.npy"
    # test_feature = extract_audio_feature(test_path, test_save_path)
    # test_feature_normalized, _ = normalize_features(test_feature, scaler=scaler, method='minmax')
    # test_preds = model.predict(test_feature_normalized)
    # write_result(test_preds)  # 保存测试结果函数需实现
