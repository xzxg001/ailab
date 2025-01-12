import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from funasr import AutoModel

import os
import numpy as np
from funasr import AutoModel

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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
def calculate_score_classification(preds, labels, average_f1='macro'):  # weighted, macro 模型预测的标签 实际的标签
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix#准确率、召回率、F1 分数、精度和混淆矩阵

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 简单的线性层

    def forward(self, x):
        return self.linear(x)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

class MyRF:
    def __init__(self):
        # 直接使用您提供的最佳参数
        self.model = SVC(
            C=0.01,
            class_weight=None,
            coef0=0.5075490730497512,
            degree=6,
            gamma='scale',
            kernel='poly',
            tol=5e-05
        )

    def train(self, features, labels):
        print("开始训练SVM模型...")
        self.model.fit(features, labels)
        print("SVM模型训练完成！")

    def evaluate(self, features, labels):
        print("开始评估模型...")
        preds = self.model.predict(features)
        accuracy, recall, f1, precision, conf_matrix = calculate_score_classification(preds, labels)
        return accuracy, recall, f1, precision, conf_matrix
    
## 读取train.csv、dev.csv
train_csv = pd.read_csv("./CSVfile/train.csv", sep = "#")
dev_csv = pd.read_csv("./CSVfile/dev.csv", sep = "#")
## 分离文件路径和标签
## 可先截取少量样本验证代码正确性，再使用所有样本
# train_path = list(train_csv.path)[:100]
# train_label = list(train_csv.label)[:100]
# dev_path = list(dev_csv.path)[:50]
# dev_label = list(dev_csv.label)[:50]

train_path = list(train_csv.path)
train_label = list(train_csv.label)
dev_path = list(dev_csv.path)
dev_label = list(dev_csv.label)
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
## 特征处理函数，可以对提取的特征进行处理，以获得更好的特征表示
def feature_process(feature):
    return feature

## 主函数
## 特征处理函数，可以对提取的特征进行处理，以获得更好的特征表示
def feature_process(feature):
    return feature

## 主函数
if __name__ == "__main__":
    ## 实例化模型
    rf = MyRF()
    ## 提取训练样本特征
    ## 文件数量很多时需要的时间较长，请耐心等待
    train_save_path="./newfeature/train_feature.npy"
    train_feature = extract_audio_feature(train_path,train_save_path) ## np.array (n,88)
    train_feature = feature_process(train_feature)
    
    ##训练模型
    rf.train(train_feature,train_label)
    ##计算在dev上的性能
    dev_save_path="./newfeature/dev_feature.npy"
    dev_feature = extract_audio_feature(dev_path,dev_save_path)
    dev_feature = feature_process(dev_feature)
    acc,ua,f1,pre,confuse_matrix = rf.evaluate(dev_feature,np.array(dev_label))
    print(f"Acc:{acc} \nUa:{ua} \nMacro_F1:{f1} \nPre:{pre}\nConfuse_matrix:\n{confuse_matrix}")
    
    ##读入test.csv
    test_csv = pd.read_csv("./CSVfile/test.csv",sep = "#")
    test_path = list(test_csv.path)
    test_save_path="./newfeature/test_feature.npy"
    test_feature = extract_audio_feature(test_path,test_save_path)
    test_label = rf.model.predict(test_feature)
    test_preds = test_label
    print(len(test_feature))
    
    ## 通过模型获得测试集对应的标签列表test_preds，并写入到result.csv文件中
    write_result(test_preds)