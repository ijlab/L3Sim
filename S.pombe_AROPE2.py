# Sample run on BlogCatalog
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
import utils
from eval import Precision_Np
import networkx as nx
import random
#AROPE的eva改动版本！！！！考虑了是环的pair，以及得分是0的pair，节点对（u,v）和（v,u）的得分只保留了位于矩阵上三角的那一个（原始版本排除了环，并且只保留了得分是正数和负数的pair，而且针对节点对u,v分别保留了u,v和v,u的得分）
def Matrix_power(Matrix,power): #Matrix是np.array格式,power是大于1的正整数
   result=Matrix
   for i in range(power-1):       
       result=np.dot(result,Matrix)
   return result

if __name__ == '__main__':

  iteration,top=10,10000
  print('Network name: S.pombe')
  dimen=127  #d的取值
  print('dimen=',dimen+1)        
  print('----------------------------------------------------------------------')
  print ('Test network G0 (the original network):')
  G0=nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_00.txt', delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')
  n=len(G0.nodes())
  nodes=[i for i in G0.nodes()] 
  np.savetxt("//extend2//chenyu//Sim//S.pombe//nodes.txt", nodes, fmt='%s')
  H0=np.zeros((n,n),dtype=np.float32)  #全部正样本
  for (u,v,w) in G0.edges(data=True):
      u1,v1=nodes.index(u),nodes.index(v)
      H0[u1,v1],H0[v1,u1]=w['weight'],w['weight']
  H2=np.zeros((n,n),dtype=np.float32)  #全部负样本
  for i in range(n):
      for j in range(n):
          if H0[i,j]==0:
             H2[i,j]=1
  A0=csr_matrix(H0)
  n=A0.shape[0]
  print('n=',n)
  print('----------------------------------------------------------------------')
  mean_precision=[]
  for kfold in range(iteration):
    t1=time.time()
    print ('Fold%d'%(kfold+1)) 
    print ('Training network G (the in-complete network):')
    G= nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe_01_%d.txt'%(kfold+1), delimiter=' ',  create_using=None, nodetype=None, encoding='utf-8')    
    H=np.zeros((n,n),dtype=np.float32)    #训练集正样本
    for (u,v,w) in G.edges(data=True):
        u1,v1=nodes.index(u),nodes.index(v)
        H[u1,v1],H[v1,u1]=w['weight'],w['weight']
    A=csr_matrix(H)
    print('----------------------------------------------------------------------')
    print('train weihts for A**2 and A**3:')
    X2,X3,X4,X5,X6=np.trace(Matrix_power(H,2)),np.trace(Matrix_power(H,3)),np.trace(Matrix_power(H,4)),np.trace(Matrix_power(H,5)),np.trace(Matrix_power(H,6))
    a = np.array([[X4,X5],[X5,X6]]) #等式左边的系数
    b = np.array([[X3],[X4]]) #等式右边的常数项
    x = np.linalg.solve(a, b)
    d,e=x[0][0],x[1][0]
    #np.allclose(np.dot(a, x), b)
    B=d*Matrix_power(H,2)+e*Matrix_power(H,3)
    #Z=B+B.T #预测矩阵
    loss=np.linalg.norm(H-B) #矩阵F范数
    print ('loss=',loss)
    print ('d,e=',d,e)
    print('----------------------------------------------------------------------')
    order = [1,2,3,-1]
    weights= [[1], [1, 0.1], [0,d,e], [0.001]]#分别表示只考虑1阶；1-2阶；1-3阶和1-无穷阶的各阶的权重
    U_list,V_list = utils.AROPE(A,dimen,order,weights,n) #节点嵌入L维向量,L根据d=dimen算出来，A特征值的top-L个中至少有d个是大于0的，这样的最小L
    U,V=U_list[0],V_list[0]
    #np.savetxt('//extend2//chenyu//Sim//S.pombe//S.pombe_01_1_U_list0.txt',U_list[0], fmt='%s')
    #np.savetxt('//extend2//chenyu//Sim//S.pombe//S.pombe_01_1_V_list0.txt',V_list[0], fmt='%s')
    #print('U=',U)
    #print('V=',V)

    a=U.dot(V.T) #矩阵U乘以V的转置,结果Sim是N*N方阵，因为U和V都是N*(d+1)矩阵.AROPE得分矩阵,得分可能有负数！！
    #print('a=',a)
    for i in range(1,n):
       for j in range(i):
          a[i,j],H0[i,j],H[i,j],H2[i,j]=0,0,0,0  #左下角元素令为0，保留主对角线
    row1, col1 = np.nonzero(H0-H) #返回测试集正样本矩阵中非零元的索引，即：测试集正样本坐标
    row2, col2 = np.nonzero(H2) #返回测试集负样本矩阵中非零元的索引，即：测试集负样本坐标 
    row_test, col_test = np.nonzero(H2+H0-H)  #全部测试集样本的坐标，H2：全部负样本，H0：全部正样本，H训练集正样本
    row, col = np.nonzero(H)  #训练集正样本的坐标
    '''for i in range(len(row)):
       a[row[i], col[i]]=0          #只保留正负测试集的得分''' 
    value=a[row_test, col_test] #全部测试机样本的得分
    '''minvalue,maxvalue=np.min(value),np.max(value) #最大和最小得分
    for i in range(len(row_test)):
        a[row_test[i], col_test[i]]=(a[row_test[i], col_test[i]]-minvalue)/(maxvalue-minvalue) #归一化，0-1之间 '''
    np.savetxt('//extend2//chenyu//Sim//S.pombe//AROPE_fold%d.txt'%(kfold+1),a, fmt='%s')   
    #AROPE=np.loadtxt('//extend2//chenyu//Sim//S.pombe//AROPE.txt') 
    value_P,value_N=a[row1, col1],a[row2, col2]     #正负测试集的AROPE得分
    P=[[value_P[i],1,row1[i], col1[i]] for i in range(len(value_P))] #存储测试集正样本L3得分，标签，节点1，节点2
    N=[[value_N[i],0,row2[i], col2[i]] for i in range(len(value_N))] #存储测试集负样本L3得分，标签，节点1，节点2
    re1=P+N
    '''temp_ele=[]
    for i in range(len(re1)):
        if re1[i][2]==re1[i][3]:
           temp_ele.append(re1[i]) #去除是环的pair
        if re1[i][0]==0:
           temp_ele.append(re1[i]) #去除得分为0的pair
    re1=[i for i in re1 if i not in temp_ele]'''
    re1.sort(reverse=True) #将综合记录的L3得分,标签，以及节点对的名称按照得分从大到小排序
    re1=re1[:top]
    np.savetxt('//extend2//chenyu//Sim//S.pombe//AROPEscore_fold%d.txt'%(kfold+1),re1,fmt='%s') #保存L3的这些信息为txt文档 
    label1=np.array([re1[i][1] for i in range(len(re1))]) #标签
    #print('re1[0],re1[1],re1[2],re1[3]=',re1[0],re1[1],re1[2],re1[3])
    precision1=np.divide(np.cumsum(label1>0), np.array(range(len(re1)))+1) 
    np.savetxt('//extend2//chenyu//Sim//S.pombe//AROPEprecision_fold%d.txt'%(kfold+1),precision1)
    mean_precision.append(precision1)
    print('Precision of Top100,Top500 and Top1000 are',precision1[99],precision1[499],precision1[999])
    print('----------------------------------------------------------------------')
    t6=time.time()
    print("fold%d time:%s"%(kfold+1,t6-t1))
  mean_P=np.mean(mean_precision,axis=0)
  print('AROPE Mean precision of Top100,Top500 and Top1000 are',mean_P[99],mean_P[499],mean_P[999])
  t6=time.time()
  print("Total time:%s"%(t6-t1))
    