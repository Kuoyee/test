# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:54:25 2021

@author: asus
"""
import numpy as np
import pandas as pd
#要列名：header=0  不要列名：header=None
#encoding='utf-8'   encoding='gbk'
df = pd.read_csv(r'/home/hadoop/代码/rent2.csv',encoding='UTF-8') 
a=0
for i in range(df.shape[0]):
    #Rentmode
    if df.iloc[i,3] =='整租':
        df.iloc[i,3] = 1
    else:
        df.iloc[i,3] = 0
    #Orientation
    if df.iloc[i,9].find('/') == -1:
        ort = df.iloc[i,9]
    else:
        ort = df.iloc[i,9].split('/')[0]
    if ort == '南':
        df.iloc[i,9] = 7
    elif ort == '东南':
        df.iloc[i,9] = 6
    elif ort == '东':
        df.iloc[i,9] = 5
    elif ort == '西南':
        df.iloc[i,9] = 4
    elif ort == '北':
        df.iloc[i,9] = 3
    elif ort == '西':
        df.iloc[i,9] = 2
    elif ort == '东北':
        df.iloc[i,9] = 1
    else:
        df.iloc[i,9] = 0
    #mantain
    if df.iloc[i,11] =='今天':
        df.iloc[i,11] = 0
    elif '天前' in df.iloc[i,11]:
        df.iloc[i,11] = int(df.iloc[i,11][:-2])
    elif '月' in df.iloc[i,11]:
        df.iloc[i,11] = int(df.iloc[i,11][:-3])*30
    #Elevator
    if df.iloc[i,12] =='有':
        df.iloc[i,12] = 1
    else:
        df.iloc[i,12] = 0
    #Parking
    if df.iloc[i,13] =='免费使用':
        df.iloc[i,13] = 2
    elif df.iloc[i,13] =='租用车位':
        df.iloc[i,13] = 1
    else:
        df.iloc[i,13] = 0
    #Gas
    if df.iloc[i,14] =='有':
        df.iloc[i,14] = 1
    else:
        df.iloc[i,14] = 0
    
df.to_csv(r'/home/hadoop/代码/rent3.csv',index=0,encoding='UTF-8')