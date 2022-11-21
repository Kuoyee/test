# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:49:31 2021

@author: asus
"""
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"#我的mpl库有点问题
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df=pd.read_csv(r"D:\p_file\course\大数据综合实习\带类别标签房屋数据.csv",encoding="utf-8")
datatotal=df.iloc[:,1:5]
labeltotal=df.iloc[:,5]-1 #从0开始
X_train,X_test,Y_train,Y_test=train_test_split(datatotal,labeltotal,test_size=0.3)
X_train = torch.tensor(np.array(X_train),dtype=torch.float32)
print(X_train.shape)
X_test = torch.tensor(np.array(X_test),dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train))
Y_test = torch.tensor(np.array(Y_test))
batch_size = 256
## 将训练数据的特征和标签组合
train_data = Data.TensorDataset(X_train, Y_train)
test_data = Data.TensorDataset(X_test, Y_test)
## 随机读取小批量
train_iter = Data.DataLoader(train_data, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_data, batch_size, shuffle=True)
for X, y in train_iter:
    print(X.shape, y.shape)
    break

#初始化模型参数
num_inputs = 4#每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是28*28=784
num_outputs = 3 #有10个输出类别

#构建前馈神经网络，其中使用relu函数作为激励函数
class ForwardNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ForwardNN, self).__init__()
        #因为前面数据返回的每个batch样本 x 的形状为(batch_size, 1, 28, 28)
        #所以要先用 view() 将 x 的形状转换成(batch_size, 784)才送入全连接层
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)  #5个卷积层
        self.fc6 = nn.Linear(16, num_outputs)  #最后一层为全连接输出层
        
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)
        y = self.fc4(y)
        y = self.relu(y)
        y = self.fc5(y)
        y = self.relu(y)
        y = self.fc6(y)
        return y

#评价模型 net 在数据集上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    data_preds = torch.tensor([])#存放所有的预测
    data_targets = torch.tensor([]) #存放所有的标签
    for X, y in data_iter:
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
        data_preds = torch.cat((data_preds, y_hat),dim=0)  #加入批量预测
        data_targets = torch.cat((data_targets, y.float().view(-1,1)),dim=0) #加入批量标签,转换成float类型之后才行
    return acc_sum / n, data_preds, data_targets

num_epochs = 20
#设置网络、损失函数、优化算法
net = ForwardNN(num_inputs, num_outputs)    
##softmax和交叉熵损失函数
##PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数
loss = nn.CrossEntropyLoss() #由于包括softmax运算，因此前馈神经网络的构建中就不用加入softmax了
##定义优化算法
##小批量随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.0015) 

#训练模型
test_preds = torch.tensor([])#存放所有的预测
test_targets = torch.tensor([]) #存放所有的标签
def train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size,params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        test_preds = torch.tensor([])
        test_targets = torch.tensor([]) 
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().item()
            n += y.shape[0]
        test_acc, test_preds, test_targets = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc)) 
    return  test_preds, test_targets  #返回所有的预测和标签，便于计算混淆矩阵
test_preds, test_targets = train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, None, None,optimizer) 

#模型评价,在测试集上的混淆矩阵
##将真实值与预测值封装
stacked = torch.stack((test_targets.long().view(-1),test_preds.argmax(dim=1)),dim=1)
print(stacked.shape)
##构建混淆矩阵
cmt = torch.zeros(3,3, dtype=torch.int64)
for p in stacked:
    tl, pl = p.tolist()  #tl代表真实标签，pl代表预测标签
    cmt[tl, pl] = cmt[tl, pl] + 1  #对于矩阵数加1
##绘制混淆矩阵热力图
names = ('CLASS:1', 'CLASS:2', 'CLASS:3',)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdPu):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.figure(figsize=(10,8))
plot_confusion_matrix(cmt, names)
##计算精确度、召回率、F1Score
reporttable = PrettyTable()#创建一个表格
reporttable.field_names = ["", "Precision", "Recall", "F1Score"]
for i in range(num_outputs):
    TP = cmt[i, i].tolist()
    FP = np.sum(cmt[i, :].tolist()) - TP
    FN = np.sum(cmt[:, i].tolist()) - TP
    TN = np.sum(cmt.tolist()) - TP - FP - FN
    Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
    Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#每一类准确度
    F1Score = round((2*Precision*Recall)/(Precision+Recall),3)
    reporttable.add_row([names[i],Precision, Recall, F1Score])
print(reporttable) 