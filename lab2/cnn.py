import opensmile
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 提取特征函数
def extract_audio_feature(file_list, save_path):
    if os.path.exists(save_path):
        print(f"已找到保存的特征文件 '{save_path}'，正在加载...")
        feature = np.load(save_path)
        print("特征加载完毕！")
        return feature
    print("请耐心等待特征提取完成！")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals)
    feature = []
    for n, file in enumerate(file_list):
        try:
            y = smile.process_file(file)
            y = y.to_numpy().reshape(-1)
            feature.append(y)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
        if (n + 1) % 100 == 0:
            print(f"当前进度 {n + 1}/{len(file_list)}")
    print("此次特征提取已结束")
    if not feature:
        raise ValueError("没有提取到任何特征，请检查音频文件路径和格式。")
    feature = np.stack(feature, axis=0)
    np.save(save_path, feature)
    print(f"特征已保存到文件 '{save_path}'")
    return feature

# 归一化特征
def normalize_features(features, scaler=None, method='minmax'):
    if method == 'minmax':
        if scaler is None:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        print("最小-最大归一化完成")
    else:
        raise ValueError("未知的归一化方法，请选择 'minmax'。")
    return features, scaler

# 计算分类评估指标
def calculate_score_classification(preds, labels, average_f1='macro'):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, recall, f1, precision, confuse_matrix

# CNN模型定义
class CNNEmotionClassifier(nn.Module):
    def __init__(self, input_dim=88, num_classes=4):
        super(CNNEmotionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.25)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(256 * (input_dim // 8), 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout1(self.pool(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool(torch.relu(self.bn3(self.conv3(x)))))
        x = x.view(-1, 256 * (x.size(2)))
        x = self.dropout4(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNModel:
    def __init__(self, input_dim=88, num_classes=4, learning_rate=0.001):
        self.model = CNNEmotionClassifier(input_dim=input_dim, num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train(self, train_loader, dev_loader, epochs=20, patience=10):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct_val_predictions = 0
            total_val_predictions = 0
            with torch.no_grad():
                for val_inputs, val_labels in dev_loader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                    val_outputs = self.model(val_inputs)
                    val_loss += self.criterion(val_outputs, val_labels).item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    correct_val_predictions += (val_predicted == val_labels).sum().item()
                    total_val_predictions += val_labels.size(0)

            val_epoch_loss = val_loss / len(dev_loader)
            val_epoch_accuracy = correct_val_predictions / total_val_predictions
            print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")
            dev_preds, dev_labels = self.evaluate(dev_loader)
            acc, ua, f1, pre, confuse_matrix = calculate_score_classification(dev_preds, dev_labels)
            print(f"dev:\nAcc:{acc} \nUa:{ua} \nMacro_F1:{f1} \nPre:{pre}\nConfuse_matrix:\n{confuse_matrix}")
            self.scheduler.step(val_epoch_loss)

            # Early stopping
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_without_improvement = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch+1} epochs without improvement.")
                    break

    def evaluate(self, loader):
        self.model.eval()
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.to(self.device)
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels

# 数据加载器创建函数
def create_dataloader(features, labels, batch_size=32, shuffle=False):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 主函数
if __name__ == "__main__":
    # 读取数据
    train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
    dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")

    train_path = list(train_csv.path)
    train_label = list(train_csv.label)
    dev_path = list(dev_csv.path)
    dev_label = list(dev_csv.label)

    # 确保标签是整数类型
    train_label = np.array(train_label).astype(int)
    dev_label = np.array(dev_label).astype(int)

    # 特征提取与归一化
    train_save_path = "./newfeature/train_feature.npy"
    train_feature = extract_audio_feature(train_path, train_save_path)
    train_feature_normalized, scaler = normalize_features(train_feature, method='minmax')
    dev_save_path = "./newfeature/dev_feature.npy"
    dev_feature = extract_audio_feature(dev_path, dev_save_path)
    dev_feature_normalized, _ = normalize_features(dev_feature, scaler=scaler, method='minmax')
    # 创建训练和验证数据加载器
    train_loader = create_dataloader(train_feature_normalized, train_label, batch_size=32)
    dev_loader = create_dataloader(dev_feature_normalized, dev_label, batch_size=32, shuffle=False)

    # 模型定义与训练
    model = CNNModel(input_dim=train_feature_normalized.shape[1], num_classes=len(np.unique(train_label)), learning_rate=0.001)
    model.train(train_loader, dev_loader, epochs=50, patience=10)

    # 训练集评估
    train_preds, train_labels = model.evaluate(train_loader)
    acc, ua, f1, pre, confuse_matrix = calculate_score_classification(train_preds, train_labels)
    print(f"train:\nAcc:{acc} \nUa:{ua} \nMacro_F1:{f1} \nPre:{pre}\nConfuse_matrix:\n{confuse_matrix}")

    # 开发集评估
