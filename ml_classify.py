# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:27:20 2021

@author: asus
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df=pd.read_csv(r"D:\p_file\course\大数据综合实习\带类别标签房屋数据.csv",encoding="utf-8")
datatotal=df.iloc[:,1:5]
labeltotal=df.iloc[:,5]
X_train,X_test,Y_train,Y_test=train_test_split(datatotal,labeltotal,test_size=0.3)

#####NBC##########################
print("朴素贝叶斯分类")
clf = GaussianNB()#默认priors=None
clf.fit(X_train, Y_train)
print("fit over")
Y_pred1=clf.predict(X_test)
print(classification_report(Y_test, Y_pred1))

####决策树#########################
print("决策树分类")
# 创建决策树对象，使用信息熵作为依据
tree_clf = tree.DecisionTreeClassifier(criterion='gini') #entropy
# fit方法分类。features为iris.data，labels为iris.target
tree_clf.fit(X_train, Y_train)
print("fit over")
Y_pred2=tree_clf.predict(X_test)
print(classification_report(Y_test, Y_pred2))


userwanted=pd.DataFrame(data=[[5,4500,100,1]],columns = ['District_id','Price','Area','Decoration'])
print(tree_clf.predict(userwanted))