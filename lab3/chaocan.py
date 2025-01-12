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
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def text_tokenize(text_list): 
    # tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased',do_lower_case=True)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
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
def preprocess_text_with_conversation_id_trans(csv_file):
    """
    从CSV中读取数据，将对话名（id字段）分别加到原始文本和翻译文本前，并计算文本长度。
    最后将原始文本和翻译文本合并到一个总的列表中。
    :param csv_file: CSV文件路径
    :return: 总的文本列表和标签列表（含长度信息）
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file, sep="#")
    
    # 处理原始文本
    data['enhanced_original_text'] = data['id'] + ": " + data['text']
    data['original_text_length'] = data['text'].apply(len)
    enhanced_original_texts = data['enhanced_original_text'] + " (" + data['original_text_length'].astype(str) + ")"
    
    # 处理翻译文本
    data['enhanced_translated_text'] = data['id'] + ": " + data['translated']
    data['translated_text_length'] = data['translated'].apply(len)
    enhanced_translated_texts = data['enhanced_translated_text'] + " (" + data['translated_text_length'].astype(str) + ")"
    
    # 将原始文本和翻译文本合并到一个总的列表中
    total_texts = list(enhanced_original_texts) + list(enhanced_translated_texts)
    
    # 标签也需要重复一次，因为每个文本（原始和翻译）都对应同一个标签
    total_labels = list(data['label']) + list(data['label'])
    
    # 返回总的文本列表和标签列表
    return total_texts, total_labels
# 处理训练和验证集
train_txt, train_label = preprocess_text_with_conversation_id_trans("./CSVfile/trans.csv")
dev_txt, dev_label = preprocess_text_with_conversation_id("./CSVfile/ans.csv")
dev1_txt, dev1_label = preprocess_text_with_conversation_id("./CSVfile/dev.csv")
test_txt = preprocess_text_with_conversation_id_test("./CSVfile/test.csv")

# 文本向量化
train_coded_txt = text_tokenize(train_txt)
dev_coded_txt = text_tokenize(dev_txt)
dev1_coded_txt = text_tokenize(dev1_txt)
test_coded_txt = text_tokenize(test_txt)

# 构建数据集
train_dataset = TensorDataset(train_coded_txt["input_ids"], 
                              train_coded_txt["attention_mask"],
                              torch.tensor(train_label))
dev_dataset = TensorDataset(dev_coded_txt["input_ids"], 
                            dev_coded_txt["attention_mask"],
                            torch.tensor(dev_label))
dev1_dataset = TensorDataset(dev1_coded_txt["input_ids"], 
                             dev1_coded_txt["attention_mask"],
                             torch.tensor(dev1_label))
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, model, device, weight_decay=0.01, num_training_steps=None, num_warmup_steps=0, lr=2e-5, alpha=1, gamma=2, params=None):
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
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

    def train(self, dataloader_train, dataloader_dev, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader_train:
                input_ids, attention_mask, labels = tuple(t.to(self.device) for t in batch)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            val_loss1, accuracy1, ua1, f11, precision, confuse_matrix1 = self.evaluate(dataloader_dev1)
            print(f"Dev1 Loss: {val_loss1:.4f}, Accuracy: {accuracy1:.4f}, F1 Score: {f11:.4f}, Precision: {precision:.4f}, confuse_matrix:\n{confuse_matrix1}")
            # 验证集评估
            val_loss, accuracy, ua, f1, precision, confuse_matrix = self.evaluate(dataloader_dev)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, confuse_matrix:\n{confuse_matrix}")
            
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
def objective(trial):
    # 定义超参数搜索空间
    
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-4)  # 学习率
    alpha = trial.suggest_uniform('alpha', 0.5, 2.0)  # Focal Loss 的 alpha
    gamma = trial.suggest_uniform('gamma', 0.5, 3.0)  # Focal Loss 的 gamma
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)  # 权重衰减
    warmup_fraction = trial.suggest_uniform('warmup_fraction', 0.05, 0.15)  # warmup 步数比例
    total_steps = len(dataloader_train) * epochs
    # 计算 num_warmup_steps
    num_warmup_steps = int(total_steps * warmup_fraction)
    
    # 超参数字典
    params = {
        'lr': lr,
        'alpha': alpha,
        'gamma': gamma,
        'weight_decay': weight_decay,
        'warmup_fraction': warmup_fraction
    }
    total_steps = len(dataloader_train) * epochs
    # 实例化模型（使用当前超参数）
    set_seeds(17)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text",trust_remote_code=True, num_labels=4, output_hidden_states=True).to(device)
    mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps, num_warmup_steps=num_warmup_steps,
                        lr=lr, alpha=alpha, gamma=gamma, weight_decay=weight_decay, params=params)
    
    # 训练模型
    mymodel.train(dataloader_train, dataloader_dev, epochs)

    # 返回负的 F1 分数（Optuna 最小化目标）
    return -mymodel.best_f1

if __name__ == "__main__":
    # 设置 Optuna 研究
    study_name = 'new'
    storage_name = 'sqlite:///{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                            direction='minimize', sampler=TPESampler(), pruner=MedianPruner())

# 开始优化
    epochs = 20
    total_steps = len(dataloader_train) * epochs
    study.optimize(objective, n_trials=3)
    
# 输出最佳超参数
    print('最佳试验结果:')
    trial = study.best_trial
    print('  最佳值 (F1 分数): ', -trial.value)
    print('  最佳超参数: ')
    best_params = trial.params
    for key, value in best_params.items():
        print('    {}: {}'.format(key, value))
        # 将最佳参数保存到 txt 文件
    with open('best_params.txt', 'w') as f:
        f.write("最佳试验结果:\n")
        f.write(f"  最佳值 (F1 分数): {-trial.value}\n")
        f.write("  最佳超参数:\n")
        for key, value in best_params.items():
            f.write(f"    {key}: {value}\n")
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        set_seeds(17)
        pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text",trust_remote_code=True, num_labels=4, output_hidden_states=True).to(device)
# 使用最佳超参数重新训练模型
    set_seeds(17)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("meghanadh/finetune_bert_iemocap_text",trust_remote_code=True, num_labels=4, output_hidden_states=True).to(device)
    mymodel = MyDLmodel(pretrained_model, device, num_training_steps=total_steps, num_warmup_steps=int(total_steps * best_params['warmup_fraction']),
                    lr=best_params['lr'], alpha=best_params['alpha'], gamma=best_params['gamma'],
                    weight_decay=best_params['weight_decay'], params=best_params)
    mymodel.train(dataloader_train, dataloader_dev, epochs=20)

# 保存模型
    torch.save(mymodel.model.state_dict(), 'best_model.pth')

# 预测测试集
    test_preds = mymodel.predict(dataloader_test)
    write_result(test_preds)
    # 计算总训练步数

    #判断在test上的效果
    test_df = pd.read_csv("./CSVfile/result.csv", sep="#")
    test_labels = test_df['label'].tolist()
    ans_df = pd.read_csv("./CSVfile/ans.csv", sep="#")
    true_labels = ans_df['label'].tolist()
    # 确保标签是整数形式
    true_labels = [int(label) for label in true_labels]
    # 计算宏平均F1分数
    f1 = f1_score(true_labels, test_labels, average='macro')
    print(f"测试集上的宏平均F1分数为: {f1:.4f}")
