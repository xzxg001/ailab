import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import numpy as np
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
from sklearn.utils.class_weight import compute_class_weight
import json
from torchview import draw_graph
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from funasr import AutoModel
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


def preprocess_text_with_conversation_id(csv_file):
    data = pd.read_csv(csv_file, sep="#")
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    
    # 返回处理后的文本和标签
    return list(enhanced_with_length), list(data['label'])

def preprocess_text_with_conversation_id_test(csv_file):
    data = pd.read_csv(csv_file, sep="#")
    data['enhanced_text'] = data['id'] + ": " + data['text']
    data['text_length'] = data['text'].apply(len)  # 计算文本长度
    enhanced_with_length = data['enhanced_text'] + " (" + data['text_length'].astype(str) + ")"
    # 返回处理后的文本和标签
    return list(enhanced_with_length)

def extract_audio_feature(file_list, save_path, granularity="utterance"):
    """
    使用 emotion2vec 提取音频特征
    :param file_list: 音频文件路径列表 (list of str)
    :param save_path: 特征保存路径 (str)
    :param granularity: 特征粒度，"utterance" 或 "frame" (str)
    :return: 提取的特征 (numpy.ndarray)
    """
    # 如果已存在保存的特征文件，则直接加载
    if os.path.exists(save_path):
        print(f"已找到保存的特征文件 '{save_path}'，正在加载...")
        feature = np.load(save_path)
        print("特征加载完毕！")
        return feature
    print("请耐心等待特征提取完！")
    # 加载 emotion2vec 模型
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(
        model=model_id,
        hub="ms",  # "ms" 或 "modelscope" 用于中国大陆用户；"hf" 或 "huggingface" 用于其他海外用户
        disable_update=True
    )
    feature = []
    for n, file in enumerate(file_list):
        # 提取特征
        rec_result = model.generate(file, output_dir="./outputs", granularity=granularity)
        # 处理返回值
        if isinstance(rec_result, list) and len(rec_result) > 0:
            feats = rec_result[0].get('feats', None)  # 获取第一个元素的 'feats'
        else:
            feats = None

        if feats is None:
            raise ValueError(f"无法从文件 {file} 中提取特征，返回值格式不正确: {rec_result}")
        # 如果是 utterance-level，特征形状为 [768]
        if granularity == "utterance":
            feats = np.array(feats).reshape(-1)  # 转换为 numpy 数组并展平
        # 如果是 frame-level，特征形状为 [T, 768]
        elif granularity == "frame":
            feats = np.array(feats)
        feature.append(feats)
        if (n + 1) % 100 == 0:
            print(f"当前进度 {n + 1}/{len(file_list)}")

    print("此次特征提取已结束")
    print("-------------------------------")
    # 将特征堆叠为 numpy 数组
    if granularity == "utterance":
        feature = np.stack(feature, axis=0)  # 形状: (len(file_list), 768)
    elif granularity == "frame":
        feature = np.array(feature)  # 形状: (len(file_list), T, 768)
    # 将特征保存到指定的文件中
    np.save(save_path, feature)
    print(f"特征已保存到文件 '{save_path}'")
    return feature

train_txt, train_label = preprocess_text_with_conversation_id("./CSVfile/train.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
dev1_txt, dev1_label = preprocess_text_with_conversation_id("./CSVfile/ans.csv")
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
dev_audio_features = extract_audio_feature(dev_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/dev_feature.npy")
dev_dataset = TensorDataset(
    dev_coded_txt["input_ids"], 
    dev_coded_txt["attention_mask"], 
    torch.tensor(dev_audio_features), 
    torch.tensor(dev_label)
)
dev1_audio_features = extract_audio_feature(dev1_path, "/old_home/lyt/zxj_workplaces/ailab/CodeWithDataset/newfeature/test_feature.npy")
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
# 定义Focal Loss
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
class CombinedModel(nn.Module):
    def __init__(self, pretrained_model, num_labels=4):
        super(CombinedModel, self).__init__()
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # Projection layer to map RoBERTa output (768) to 1024 dimensions
        self.text_projection = nn.Linear(768, 1024)
        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Max pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Fully connected layer after CNNs
        self.fc = nn.Linear(256 * 512, 768)  # Adjust based on the output size of the last CNN layer
        # Dropout layer
        self.dropout = nn.Dropout(0.1)       
        # Classification layer
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, audio_features):
        # Get RoBERTa output
        roberta_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)        
        # 获取最后一层的隐藏状态（768 维）
        last_hidden_state = roberta_output.hidden_states[-1]  # shape: (batch_size, sequence_length, 768)
        # 使用 [CLS] 位置的隐藏状态作为文本特征
        text_features = last_hidden_state[:, 0, :]  # shape: (batch_size, 768)
        
        # Project text features to 1024 dimensions
        text_features = self.text_projection(text_features)  # Shape: (batch_size, 1024)
        
        # Combine text and audio features
        combined_features = torch.cat((text_features, audio_features), dim=1)  # Shape: (batch_size, 2048)
        
        # Reshape combined features for CNN input
        combined_features = combined_features.unsqueeze(1)  # Shape: (batch_size, 1, 2048)
        
        # First CNN layer
        x = F.relu(self.conv1(combined_features))  # Shape: (batch_size, 64, 2048)
        x = self.pool(x)  # Shape: (batch_size, 64, 1024)
        
        # Second CNN layer
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 128, 1024)
        x = self.pool(x)  # Shape: (batch_size, 128, 512)
        # Third CNN layer
        x = F.relu(self.conv3(x))  # Shape: (batch_size, 256, 512)
        # Removed last pooling layer to maintain shape (batch_size, 256, 512)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 256 * 512)
    
        x = F.relu(self.fc(x))  # Shape: (batch_size, 768)
        x = self.dropout(x)
        logits = self.classifier(x)  # Shape: (batch_size, num_labels)
        
        return logits
# 定义MyDLmodel模型类，包含逐步解冻策略
class MyDLmodel:
    def __init__(self, model, device, weight_decay=0.01, num_training_steps=None, num_warmup_steps=0, lr=2e-5, alpha=1, gamma=2, params=None):
        self.model = model
        self.model.to(device)
        self.device = device
        
        # 定义优化器，只有新的全连接层和分类层的参数会更新
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if 'pretrained_model' not in n], 'weight_decay': weight_decay}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, weights=weights_tensor)
        self.params = params  # 当前超参数
        self.best_f1 = 0.0  # 最佳F1分数

    def freeze_layers(self, num_layers_to_freeze):
        """
        冻结模型的前 num_layers_to_freeze 层
        """
        for name, param in self.model.named_parameters():
            if 'pretrained_model.encoder.layer' in name:
                layer_num = int(name.split('.')[3])
                if layer_num < num_layers_to_freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def unfreeze_layers(self, num_layers_to_unfreeze):
        """
        解冻模型的后 num_layers_to_unfreeze 层
        """
        for name, param in self.model.named_parameters():
            if 'pretrained_model.encoder.layer' in name:
                layer_num = int(name.split('.')[3])
                total_layers = 12  # 假设总共有12层，如果层数发生变化，记得修改这个值
                if layer_num >= total_layers - num_layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def train(self, dataloader_train, dataloader_dev, epochs, freeze_at_epoch=36):
        for epoch in range(epochs):
            # 逐步解冻
            if epoch >= freeze_at_epoch:
                self.unfreeze_layers(12)  # 解冻12层
            elif epoch == 0:
                self.freeze_layers(12)  # 初始冻结6层
            elif epoch <=freeze_at_epoch/3:
                self.unfreeze_layers(3)  # 中间冻结9层
            elif epoch>freeze_at_epoch/3 and epoch <=freeze_at_epoch*2/3:
                self.unfreeze_layers(6)  # 中间冻结6层
            else:
                self.unfreeze_layers(9)  # 解冻所有层
            self.model.train()
            total_loss = 0
            for batch in dataloader_train:
                input_ids, attention_mask, audio_features, labels = tuple(t.to(self.device) for t in batch)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, audio_features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Confusion Matrix:\n{confuse_matrix}")
            val1_loss, val1_accuracy, val1_ua, val1_f1, val1_precision, val1_confuse_matrix = self.evaluate(dataloader_dev1)
            print(f"Epoch {epoch+1}: Validation1 Loss: {val1_loss:.4f}, Accuracy: {val1_accuracy:.4f}, F1 Score: {val1_f1:.4f}, Precision: {val1_precision:.4f}, Confusion Matrix:\n{val1_confuse_matrix}")
    def evaluate(self, dataloader_val):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = F.cross_entropy
        for batch in dataloader_val:
            input_ids, attention_mask, audio_features, labels = tuple(t.to(self.device) for t in batch)
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
        accuracy, ua, f1, precision, confuse_matrix = self.calculate_score_classification(all_preds, all_labels)
        return avg_loss, accuracy, ua, f1, precision, confuse_matrix

    def calculate_score_classification(self, preds, labels):
        preds = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        ua = recall_score(labels, preds, average='macro', zero_division=0)
        confuse_matrix = confusion_matrix(labels, preds)
        return accuracy, ua, f1, precision, confuse_matrix

    def predict(self, dataloader_test):
        self.model.eval()
        all_preds = []
        for batch in dataloader_test:
            input_ids, attention_mask, audio_features = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, audio_features)
                logits = outputs
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
# 使用模型训练
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 训练数据和模型初始化
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text", num_labels=4, output_hidden_states=True)
    model = CombinedModel(pretrained_model, num_labels=4).to(device)
    
    epochs = 32
    total_steps = len(dataloader_train) * epochs
    all_warmup_steps = int(total_steps * 0.1)

    # 初始化模型训练类
    mymodel = MyDLmodel(model, device, num_training_steps=total_steps, num_warmup_steps=all_warmup_steps, lr=2e-5)
    mymodel.train(dataloader_train, dataloader_dev, epochs, freeze_at_epoch=3)

    # 预测
    test_preds = mymodel.predict(dataloader_test)
    write_result(test_preds)
    # 判断在test上的效果
    test_df = pd.read_csv("./CSVfile/result.csv", sep="#")
    test_labels = test_df['label'].tolist()
    ans_df = pd.read_csv("./CSVfile/ans.csv", sep="#")
    true_labels = ans_df['label'].tolist()
    # 确保标签是整数形式
    true_labels = [int(label) for label in true_labels]
    # 计算宏平均F1分数
    f1 = f1_score(true_labels, test_labels, average='macro')
    print(f"测试集上的宏平均F1分数为: {f1:.4f}")