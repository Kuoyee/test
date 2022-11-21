from math import sqrt
import operator
import numpy as np
import pandas as pd
#1.构建用户-->物品的倒排
def loadData(files):
    data ={};
    for line in files:
        user,score,item=line.split(",");
        data.setdefault(user,{});
        data[user][item]=score;
    print ("----1.用户：物品的倒排----")
    print (data)
    return (data)

#2.计算
# 2.1 构造物品-->物品的共现矩阵
# 2.2 计算物品与物品的相似矩阵
def similarity(data):
    # 2.1 构造物品：物品的共现矩阵
    N={};#喜欢物品i的总人数
    C={};#喜欢物品i也喜欢物品j的人数
    for user,item in data.items():
        for i,score in item.items():
            N.setdefault(i,0);
            N[i]+=1;
            C.setdefault(i,{});
            for j,scores in item.items():
                if j not in i:
                    C[i].setdefault(j,0);
                    C[i][j]+=1;

    print ("---2.构造的共现矩阵---")
    print ('N:',N);
    print ('C',C);

    #2.2 计算物品与物品的相似矩阵
    W={};
    for i,item in C.items():
        W.setdefault(i,{});
        for j,item2 in item.items():
            W[i].setdefault(j,0);
            W[i][j]=C[i][j]/sqrt(N[i]*N[j]);
    print ("---3.构造的相似矩阵---")
    print (W)
    return (W)

#3.根据用户的历史记录，给用户推荐物品
def recdList(data,W,user,k=3,N=10):
    rank={};
    lista=[];
    for i,score in data[user].items():#获得用户user历史记录，如A用户的历史记录为{'a': '1', 'b': '1', 'd': '1'}
        for j,w in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:k]:#获得与物品i相似的k个物品
            if j not in data[user].keys():#该相似的物品不在用户user的记录里
                rank.setdefault(j,0);
                rank[j]+=float(score) * w;
    print ("---4.推荐----")
    print (sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N])
    lista=sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]
    output = open(r'rent3.csv','w',encoding='gbk')
    output.write('ID\t\感兴趣\n')
    # print(len(lista))
    for i in range(len(lista)):
    	for j in range(len(lista[i])):
    		output.write(str(lista[i][j]))    #write函数不能写int类型的参数，所以使用str()转化
    		output.write('\t')   #相当于Tab一下，换一个单元格
    	output.write('\n')       #写完一行立马换行
    output.close()    
    return (sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N])
    #print(len(lista))
    #print(len(lista[1][1]))
    

if __name__=='__main__':

    df=pd.read_csv(r'D:\p_file\course\大数据综合实习\USER.csv',encoding='utf-8')
    print(df.columns)
    lista=[]
    for i in range(df.shape[0]):
        stringa=str(df.iloc[i,0])+','+str(df.iloc[i,2])+','+str(df.iloc[i,1])
        lista.append(stringa)

    data=loadData(lista);#获得数据
    mix=similarity(data);#计算物品相似矩阵
    re=recdList(data,mix,'User',3,3);#推荐'C:\Users\tc\Desktop\123.txt'
