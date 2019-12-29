# -*- coding: utf-8 -*-
#from __future__ import division 
import numpy as np
import networkx as nx
import time
import os
from numba import cuda,jit
from copy import deepcopy
########################
print ('Network: S.pombe')
iteration,top=10,10000
###################
@cuda.jit
def DEG(X,D,n):
  i=cuda.grid(1)
  d=0
  for z in range(n):
    if X[i,z]!=0:
      d+=1
  D[i]=d
    
@cuda.jit
def cal(D,X,X1,X2,XA1,XA2,XB1,XB2):
  i,j=cuda.grid(2)
  #print(i,j)

  if X1[i,j]!=0:            #has edge 
    aa,bb=0,0 
    listj=X[j,:]
    for v in range(len(listj)):
      if listj[v]!=0 and v!=i:      # is neighbor
        a,b,wei2=0,0,X[v,j]        
        listi=X[i,:]
        for u in range(len(listi)):
          if listi[u]!=0 and u!=j:    # is neighbor
            wei3=X[u,i]
            if X[u,v]!=0:
              wei4=X[u,v]
              a+=(wei2*wei3*wei4)/(D[u]**0.5*D[v]**0.5*D[i]**0.5*D[j]**0.5)
              #print(i,j,a)
              b+=(wei2*wei3*wei4)/(D[u]**0.5*D[v]**0.5)
        aa+=a
        bb+=b
    XA1[i,j]=aa
    #print(aa)
    XB1[i,j]=bb
   # print(bb)

  if X2[i,j]!=0:            #has edge 
    aa,bb=0,0
    listj=X[j,:]
    for v in range(len(listj)):
      if listj[v]!=0 and v!=i:      # is neighbor
        a,b,wei2=0,0,X[v,j]
        listi=X[i,:]
        for u in range(len(listi)):
          if listi[u]!=0 and u!=j:    # is neighbor
            wei3=X[u,i]
            if X[u,v]!=0:
              wei4=X[u,v]
              a+=(wei2*wei3*wei4)/(D[u]**0.5*D[v]**0.5*D[i]**0.5*D[j]**0.5)
              b+=(wei2*wei3*wei4)/(D[u]**0.5*D[v]**0.5)
        aa+=a
        bb+=b
    XA2[i,j]=aa
  #  print('2:=========',i,j,aa)
    XB2[i,j]=bb
   # print('2-2:-----',i,j,bb)                 
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

mean_precision=[] #存储每一折的precision
mean_precisionX=[] #存储每一折的precision
for kfold in range(iteration):  
  t2=time.time() 
  print ('Fold%d'%(kfold+1)) 
  G= nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_01_%d.txt'%(kfold+1), delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')    
  X=np.zeros((n,n),dtype=np.float32)    #训练集正样本
  for (u,v,w) in G.edges(data=True):
      u1,v1=nodes.index(u),nodes.index(v)
      X[u1,v1],X[v1,u1]=w['weight'],w['weight']
  #X1=X1-X #测试集正样本，矩阵格式
  XA1=np.zeros(shape=(n,n),dtype=np.float32)
  XB1=np.zeros(shape=(n,n),dtype=np.float32)
  XB2=np.zeros(shape=(n,n),dtype=np.float32)
  XA2=np.zeros(shape=(n,n),dtype=np.float32)
  D=np.zeros(n,dtype=np.int32)
###################################################################################
  griddim=[n,n]
  blockdim=1
  DEG[n,blockdim](X,D,n)
  #print(D,'\n',X,'\n',X1,'\n',X2)
  cal[griddim,blockdim](D,X,X1,X2,XA1,XA2,XB1,XB2)
  #print(XA1,XA2,XB1,XB2)
############################################################################### 
  for i in range(1,n):
      for j in range(i):
          XA1[i,j],XA2[i,j],XB1[i,j],XB2[i,j],X[i,j],X1[i,j],X2[i,j]=0,0,0,0,0,0,0 #左下角元素令为0，保留主对角线   
  row1, col1 = np.nonzero(X1-X) #返回测试集正样本矩阵中非零元的索引，即：测试集正样本坐标
  row2, col2 = np.nonzero(X2) #返回测试集负样本矩阵中非零元的索引，即：测试集负样本坐标
############################################################################### 
  L3=XB1+XB2 #所有样本(包括测试集和训练集正负样本)的得分矩阵（上三角矩阵，保留主对角线）
  L3X=XA1+XA2  
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3_fold%d.txt'%(kfold+1),L3, fmt='%s')   
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3X_fold%d.txt'%(kfold+1),L3X, fmt='%s')   
  row_test, col_test = np.nonzero(X1+X2-X) #全部测试集样本的坐标
  value=L3[row_test, col_test] #全部测试集样本的得分
  valueX=L3X[row_test, col_test] #全部测试集样本的得分
  '''minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
  for i in range(len(row_test)):
      L3[row_test[i], col_test[i]]=(L3[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化 '''
  value_P,value_N=L3[row1, col1],L3[row2, col2]    
  valueX_P,valueX_N=L3X[row1, col1],L3X[row2, col2] 
  P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本L3得分，标签，节点1，节点2
  N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本L3得分，标签，节点1，节点2
  PX=[[valueX_P[i],1,row1[i], col1[i]] for i in range(len(valueX_P))] #存储测试集正样本L3得分，标签，节点1，节点2
  NX=[[valueX_N[i],0,row2[i], col2[i]] for i in range(len(valueX_N))] #存储测试集负样本L3得分，标签，节点1，节点2
  re1=P+N
  re1.sort(reverse=True) #将综合记录的L3得分,标签，以及节点对的名称按照得分从大到小排序
  re1=re1[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3score_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存L3的这些信息为txt文档 
  re2=PX+NX
  re2.sort(reverse=True) #将综合记录的L3得分,标签，以及节点对的名称按照得分从大到小排序
  re2=re2[:top]
  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3Xscore_fold%d.txt'%(kfold+1),re2,fmt='%s') #保存L3的这些信息为txt文档 

############################################################################### 
  label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
  precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 

  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3precision_fold%d.txt'%(kfold+1),precision1)
  mean_precision.append(precision1)
  print('L3 Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])

  label2=np.array([re2[i][1] for i in range(len(re2))]) #标签
  precision2=np.divide(np.cumsum(label2>0), np.array(range(len(re2)))+1) 

  np.savetxt('//extend2//chenyu//Sim//S.pombe//L3Xprecision_fold%d.txt'%(kfold+1),precision2)
  mean_precisionX.append(precision2)
  print('L3X Precision of Top100,Top500 and Top1000 are',precision2[99],precision2[499],precision2[999])

  t6=time.time()
  print("fold%d time:%s"%(kfold+1,t6-t2))
################################################################### 
mean_P=np.mean(mean_precision,axis=0)
print('L3 Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])

mean_P=np.mean(mean_precisionX,axis=0)
print('L3X Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])


P_num,N_num=len(row1),len(row2) #正样本数目
print ('#positive sample, #all potential node pairs(#all samples)=',P_num,',', P_num+N_num) #正样本数目,潜在的节点对的数目（除了训练集以外的节点对数目）
print ('proportion of positive samples in all samples=','%.8f%%' % ((P_num/(P_num+N_num))*100))    #正样本在所有样本中的比例
t6=time.time()
print("Total time:%s"%(t6-t1))

