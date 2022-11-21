# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:02:47 2021

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

num_outputs = 3
class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Sequential(
                #C1卷积层
                nn.Conv2d(1, 6, (1,2),padding=1)) # in_channels, out_channels,kernel_size)
        self.conv2 = nn.Sequential(nn.Sigmoid())
        self.conv3 = nn.Sequential(nn.MaxPool2d((1,2), 2))
        self.conv4 = nn.Sequential(nn.Conv2d(6, 16, (1,2)))
        self.conv5 = nn.Sequential(nn.Sigmoid())
        self.fc = nn.Sequential(
                nn.Linear(16*2*1, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
                )
    def forward(self, img):
        img = torch.unsqueeze(img,dim = 1)
        img = torch.unsqueeze(img,dim = 1)
        print("0",img.shape) #torch.Size([256, 1, 28, 28])
        feature1 = self.conv1(img)#torch.Size([256, 6, 24, 24])
        print("1",feature1.shape)
        feature2 = self.conv2(feature1) #torch.Size([256, 6, 24, 24])
        print(feature2.shape)
        feature3 = self.conv3(feature2) #torch.Size([256, 6, 12, 12])
        print(feature3.shape)
        feature4 = self.conv4(feature3) #torch.Size([256, 16, 8, 8])
        print(feature4.shape)
        feature5 = self.conv5(feature4) #torch.Size([256, 16, 8, 8])
        print(feature5.shape)
        output = self.fc(feature5.view(img.shape[0], -1))
        return output
#LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                #C1卷积层
                #LeNet5中输入为32*32的图片，但是mnist图片为28*28，因此两边加入填充2
                #torch.Size([256, 1, 1, 4])
                nn.Conv2d(1, 6, (1,2), padding=1), # in_channels, out_channels,kernel_size
                nn.ReLU(), #torch.Size([256, 6, 3, 5])
                #S2池化层
                nn.MaxPool2d((1,2), 2), #torch.Size([256, 6, 2, 2])
                #C3卷积层
                nn.Conv2d(6, 16, (1,2)),#torch.Size([256, 16, 2, 1])
                nn.ReLU(), #torch.Size([256, 16, 2, 1])
                )
        self.fc = nn.Sequential(
                nn.Linear(16*2*1, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
                )
    def forward(self, img):
        img = torch.unsqueeze(img,dim = 1)
        img = torch.unsqueeze(img,dim = 1)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
def evaluate_accuracy(data_iter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    data_preds = torch.tensor([]).to(device)#存放所有的预测
    data_targets = torch.tensor([]).to(device) #存放所有的标签
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                y_hat = net(X.to(device))
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    y_hat = net(X, is_training=False)
                    acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
                else:
                    y_hat = net(X)
                    acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            data_preds = torch.cat((data_preds, y_hat),dim=0)  #加入批量预测
            data_targets = torch.cat((data_targets, y.float().view(-1,1).to(device)),dim=0) #加入批量标签,转换成float类型之后才行
    return acc_sum / n, data_preds, data_targets
test_preds = torch.tensor([]).to(device)#存放所有的预测
test_targets = torch.tensor([]).to(device) #存放所有的标签
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        test_preds = torch.tensor([]).to(device)
        test_targets = torch.tensor([]).to(device)
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc, test_preds, test_targets = evaluate_accuracy(test_iter, net)    
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    return  test_preds, test_targets  #返回所有的预测和标签，便于计算混淆矩阵
##学习率采用0.001，训练算法使用Adam算法，损失函数使用交叉熵损失函数
net = LeNet()
lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
test_preds, test_targets = train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
