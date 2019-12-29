# -*- coding: utf-8 -*-
from __future__ import division 
import time
import networkx as nx
import numpy as np



print('Network: S.pombe')  #网络名称
iteration,top=10,10000

t1=time.time()
G0=nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_00.txt', delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')
nodes=[i for i in G0.nodes()] 
n=len(nodes)
print('n=',n)
X1=np.zeros((n,n),dtype=np.float32)  #全部正样本
for (u,v,w) in G0.edges(data=True):
    u1,v1=nodes.index(u),nodes.index(v)
    X1[u1,v1],X1[v1,u1]=w['weight'],w['weight']
X2=np.zeros((n,n),dtype=np.float32)  #全部负样本
for i in range(n):
    for j in range(n):
        if X1[i,j]==0:
           X2[i,j]=1
mean_precision1,mean_precision2,mean_precision3,mean_precision4,mean_precision5,mean_precision6,mean_precision7,mean_precision8,mean_precision9,mean_precision10,mean_precision11=[],[],[],[],[],[],[],[],[],[],[], 
for kfold in range(iteration):  
  print ('Fold%d'%(kfold+1))
  t2=time.time()   
  G= nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_01_%d.txt'%(kfold+1), delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')    
  X=np.zeros((n,n),dtype=np.float32)    #训练集正样本
  for (u,v,w) in G.edges(data=True):
      u1,v1=nodes.index(u),nodes.index(v)
      X[u1,v1],X[v1,u1]=w['weight'],w['weight']
  #X1=X1-X #测试集正样本，矩阵格式
  L3=np.loadtxt('//extend2//chenyu//Sim//S.pombe//L3_fold%d.txt'%(kfold+1)) 
  L3X=np.loadtxt('//extend2//chenyu//Sim//S.pombe//L3X_fold%d.txt'%(kfold+1)) 
  Sim=np.loadtxt('//extend2//chenyu//Sim//S.pombe//Sim_fold%d.txt'%(kfold+1)) 
  AROPE=np.loadtxt('//extend2//chenyu//Sim//S.pombe//AROPE_fold%d.txt'%(kfold+1)) 

  for i in range(1,n):
      for j in range(i):
         X[i,j],X1[i,j],X2[i,j]=0,0,0 #左下角元素令为0，保留主对角线   
  row1, col1 = np.nonzero(X1-X) #返回测试集正样本矩阵中非零元的索引，即：测试集正样本坐标
  row2, col2 = np.nonzero(X2) #返回测试集负样本矩阵中非零元的索引，即：测试集负样本坐标
  row_test, col_test = np.nonzero(X1+X2-X) #全部测试集样本的坐标  
  test_num=len(row_test)       
###################################################################   
  value=L3[row_test, col_test] #全部测试集样本的得分
  minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
  for i in range(test_num):
      L3[row_test[i], col_test[i]]=(L3[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化

  value=L3X[row_test, col_test] #全部测试集样本的得分
  minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
  for i in range(test_num):
      L3X[row_test[i], col_test[i]]=(L3X[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化
      
  value=Sim[row_test, col_test] #全部测试集样本的得分
  minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
  for i in range(test_num):
      Sim[row_test[i], col_test[i]]=(Sim[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化

  value=AROPE[row_test, col_test] #全部测试集样本的得分
  minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
  for i in range(test_num):
      AROPE[row_test[i], col_test[i]]=(AROPE[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化
###################################################################       
  Z=L3+L3X+Sim+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #排序得分，从大到小是True，反之是False
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+Sim+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+Sim+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision1.append(precision1)
  print('Rank of L3+L3X+Sim+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3X+Sim+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+Sim+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+Sim+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision2.append(precision1)
  print('Rank of L3X+Sim+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+Sim+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+Sim+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+Sim+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision3.append(precision1)
  print('Rank of L3+Sim+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+L3X+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision4.append(precision1)
  print('Rank of L3+L3X+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+L3X+Sim #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+Simscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3X+Simprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision5.append(precision1)
  print('Rank of L3+L3X+Sim Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+L3X #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3Xscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+L3Xprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision6.append(precision1)
  print('Rank of L3+L3X Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+Sim #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+Simscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+Simprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision7.append(precision1)
  print('Rank of L3+Sim Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision8.append(precision1)
  print('Rank of L3+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3X+Sim #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+Simscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+Simprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision9.append(precision1)
  print('Rank of L3X+Sim Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=L3X+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision10.append(precision1)
  print('Rank of L3X+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  Z=Sim+AROPE #预测矩阵，测试集位置元素更改为各自在每种方法种的rank之和
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将排序得分从小到大排列
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//Sim+AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存这些信息为txt文档 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//Sim+AROPEprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision11.append(precision1)
  print('Rank of Sim+AROPE Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
################################################################### 
  t6=time.time()
  print("fold%d time:%s"%(kfold+1,t6-t2))
################################################################### 
mean_P=np.mean(mean_precision1,axis=0)
print('L3+L3X+Sim+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision2,axis=0)
print('L3X+Sim+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision3,axis=0)
print('L3+Sim+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision4,axis=0)
print('L3+L3X+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision5,axis=0)
print('L3+L3X+Sim Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision6,axis=0)
print('L3+L3X Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision7,axis=0)
print('L3+Sim Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision8,axis=0)
print('L3+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision9,axis=0)
print('L3X+Sim Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision10,axis=0)
print('L3X+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
mean_P=np.mean(mean_precision11,axis=0)
print('Sim+AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])

#打印正样本数目以及在所有样本中的比例
P_num,N_num=len(row1),len(row2) #正样本数目
print ('#positive sample, #all potential node pairs(#all samples)=',P_num,',', P_num+N_num) #正样本数目,潜在的节点对的数目（除了训练集以外的节点对数目）
print ('proportion of positive samples in all samples=','%.8f%%' % ((P_num/(P_num+N_num))*100))    #正样本在所有样本中的比例
t6=time.time()
print("Total time:%s"%(t6-t1))

