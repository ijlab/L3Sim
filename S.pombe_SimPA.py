# -*- coding: utf-8 -*-
from __future__ import division 
import numpy as np
import networkx as nx
import time
from numba import cuda,jit
from copy import deepcopy
###############################################################################
print('Network: S.pombe')  #网络名称
iteration,top=10,10000
print('-------------------------------------------------------------------')

@cuda.jit
def DEG(X,D,n):
  i=cuda.grid(1)
  d=0
  for z in range(n):
    if X[i,z]!=0:
      d+=1
  D[i]=d

@cuda.jit
def loop(D,X,Y):
  li,lj=cuda.grid(2)
  J=D[li]*D[lj]  #两个节点的度数乘积
  Y[lj,li],Y[li,lj]=J,J
###############################################################################
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

mean_precision=[] 
for kfold in range(iteration):  
  print ('Fold%d'%(kfold+1))
  t2=time.time()   
  G= nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_01_%d.txt'%(kfold+1), delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')    
  X=np.zeros((n,n),dtype=np.float32)    #训练集正样本
  for (u,v,w) in G.edges(data=True):
      u1,v1=nodes.index(u),nodes.index(v)
      X[u1,v1],X[v1,u1]=w['weight'],w['weight']
  #X1=X1-X #测试集正样本，矩阵格式
############################################################################### 
  Y=np.zeros((n,n),dtype=np.float32)  
  D=np.zeros(n,dtype=np.int32)
  griddim=[n,n]
  blockdim=1
  DEG[n,blockdim](X,D,n)
  loop[griddim,blockdim](D,X,Y)
  Z=np.dot(X,Y)+np.dot(Y,X)   
  for i in range(1,n):
      for j in range(i):
         Z[i,j],X[i,j],X1[i,j],X2[i,j]=0,0,0,0 #左下角元素令为0，保留主对角线   
  np.savetxt('//extend2//chenyu//Sim//S.pombe//SimPA_fold%d.txt'%(kfold+1),Z, fmt='%s')   
  row1, col1 = np.nonzero(X1-X) #返回测试集正样本矩阵中非零元的索引，即：测试集正样本坐标
  row2, col2 = np.nonzero(X2) #返回测试集负样本矩阵中非零元的索引，即：测试集负样本坐标
  row_test, col_test = np.nonzero(X1+X2-X) #全部测试集样本的坐标
  value=Z[row_test, col_test] #全部测试集样本的得分
  value_P,value_N=Z[row1, col1],Z[row2, col2]  
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本Sim得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本Sim得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将综合记录的Sim得分,标签，以及节点对的名称按照得分从大到小排序
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//SimPAscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存Sim的这些信息为txt文档 

############################################################################### 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 

  np.savetxt('//extend2//chenyu//Sim//S.pombe//SimPAprecision_fold%d.txt'%(kfold+1),precision1)
  mean_precision.append(precision1)
  print('SimPA Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
  t6=time.time()
  print("fold%d time:%s"%(kfold+1,t6-t2))
################################################################### 
mean_P=np.mean(mean_precision,axis=0)
print('SimPA Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
#打印正样本数目以及在所有样本中的比例
P_num,N_num=len(row1),len(row2) #正样本数目
print ('#positive sample, #all potential node pairs(#all samples)=',P_num,',', P_num+N_num) #正样本数目,潜在的节点对的数目（除了训练集以外的节点对数目）
print ('proportion of positive samples in all samples=','%.8f%%' % ((P_num/(P_num+N_num))*100))    #正样本在所有样本中的比例
t6=time.time()
print("Total time:%s"%(t6-t1))
