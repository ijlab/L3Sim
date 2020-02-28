# -*- coding: utf-8 -*-
from __future__ import division 
import numpy as np
import networkx as nx
import time
from numba import cuda,jit
from copy import deepcopy
import sys

name=sys.argv[1] #外部命令输出方式为：python3 jacc.py name
print ('Network: %s'%name) #寻找杰卡德相似度最高的节点对
###############################################################################
t1=time.time()
G0= nx.read_weighted_edgelist('%s.txt'%name, delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8')
nodes=[i for i in G0.nodes()] #复制G的节点集为列表格式，和原节点集中节点顺序一致
n=len(nodes)
print ('num of nodes:',n)
H0=np.zeros((n,n),dtype=np.float32)  #邻接矩阵
for (u,v,w) in G0.edges(data=True):
    u1,v1=nodes.index(u),nodes.index(v)
    H0[u1,v1],H0[v1,u1]=w['weight'],w['weight']

###############################################################################
def jacc2(a, b):  #定义两个向量（邻接向量）间的相似性度量
  k1,k2=0,0 #k1与共同邻居有关，k2与非共同邻居有关
  for i in range(len(a)):
    j=a[i]*b[i]
    if j!=0:
       k1+=j
       k2+=j
    else:
       k2+=a[i]+b[i]
  if k2!=0: 
     x=k1/k2  
  else:  #如果两个向量都是零向量
     x=0 #无论规定什么值都可以，因为对链接预测没有作用
  return x 

@cuda.jit
def loop(X,Y):
  li,lj=cuda.grid(2)
  k1,k2=np.int32(0),np.int32(0) 
  for i in range(len(X[li])):
    j=X[li][i]*X[lj][i]
    if j!=0:
       k1+=j
       k2+=j
    else:
       k2+=X[li][i]+X[lj][i]
  if k2!=0:
    J=k1/k2
  else: 
    J=0
  Y[lj,li],Y[li,lj]=J,J

@cuda.jit
def DEG(X,D,n):
  i=cuda.grid(1)
  d=0
  for z in range(n):
    if X[i,z]!=0:
      d+=1
  D[i]=d

###############################################################################
Y=np.zeros((n,n),dtype=np.float32)  
D=np.zeros(n,dtype=np.int32)
griddim=[n,n]
blockdim=1
loop[griddim,blockdim](H0,Y)
DEG[n,blockdim](H0,D,n)
###############################################################################
for i in range(n):
    for j in range(i,n):
        Y[i,j]=0           #上三角和主对角线令为0
temp_row, temp_col = np.nonzero(Y)
temp_value = Y[temp_row,temp_col] 
temp_choose = np.logical_and(temp_row != temp_col, temp_row != temp_col)#非主对角线
temp_row, temp_col, temp_value = temp_row[temp_choose], temp_col[temp_choose], temp_value[temp_choose]

temp_index = np.argsort(temp_value)[::-1] #数值按从大到小排列，并返回其在temp_value中的索引
top=min(50000,len(temp_row))
temp_index = temp_index[: top] #截断取Y中非零元前top大的那些元素在temp_value中的索引（和在temp_row和 temp_col中的索引一样）
temp_value=list(temp_value)
temp_value.sort(reverse=True)
temp_value = temp_value[: top]
temp_row, temp_col = temp_row[temp_index], temp_col[temp_index] #截断取Y中非零元前int(Np)+1大的那些元素在Sim矩阵中的行标和列标

max_pairs=[]
for i in range(top):
  #if D[temp_row[i]]+D[temp_col[i]]>4: 
    max_pairs.append([nodes[temp_row[i]],nodes[temp_col[i]],temp_value[i],D[temp_row[i]]+D[temp_col[i]],])
#print('Jaccard score, sum of degrees, node A, node B=', max_pairs)              
np.savetxt('jacc%s.txt'%name,max_pairs, fmt='%s')   
#nx.draw(G, with_labels=True)   
t2=time.time()
print("time:%s"%(t2-t1))

