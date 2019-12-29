# -*- coding: utf-8 -*-
from __future__ import division 
import networkx as nx
from copy import deepcopy
import random,time

t1=time.time()
###############################################################################
print ('Network: ego-facebook')
iteration,ratio=10,0.1
###############################################################################
#十折交叉验证的样本采集过程，如果已经做了可以跳过
H=nx.read_weighted_edgelist('//extend2//chenyu//Sim//ego-facebook//ego-facebook.txt') #读取原图
G=deepcopy(H)#深复制H为G，待会儿删除G中的边可能会使得节点数变少
###############################################################################
'''#去除环
for i in G.nodes():
    if (i,i) in G.edges:
        print ('ring',(i,i)) #打印环
        G.remove_edge(i,i) #去除环'''
###############################################################################
print ('nx.info(G)')
print (nx.info(G))
#print ('average clustering coefficient:', nx.average_clustering(G))
nx.write_weighted_edgelist(G, '//extend2//chenyu//Sim//ego-facebook//ego-facebook_00.txt')#保存原图
samplesize=int(ratio*len(G.edges))  #正测试集总规模=samplesize
print ('iteration=',iteration)
edges=[]
for i in range(iteration):    
  G0=deepcopy(H)  #初始化训练集为原图    
  edges.append(random.sample(G.edges(), samplesize)) #前几轮删去边以后的图G中随机取samplesize条边  
  G.remove_edges_from(edges[i]) #删除这一轮选择的10%的边，G的边越来越少  
  G0.remove_edges_from(edges[i]) #删除这一轮选择的10%的边，G0只删去了本轮选取的samplesize条边
  nx.write_weighted_edgelist(G0, '//extend2//chenyu//Sim//ego-facebook//ego-facebook_01_%d.txt'%(i+1)) 
  #训练集（每条边都有权重）
  G1=nx.empty_graph(H.nodes()) #初始化一轮的正测试集为空图
  G1.add_edges_from(edges[i])  #本轮选取的samplesize条边为正样本集  
  nx.write_weighted_edgelist(G1, '//extend2//chenyu//Sim//ego-facebook//ego-facebook_02_%d.txt'%(i+1))
  #正样本（边都没有权重） 
  print ('positive samplesize:', len(G1.edges))
  print ('training samplesize:', len(G0.edges))
  #print (nx.info(G))
print ('sample phase is done')
t2=time.time()
print("time:%s"%(t2-t1))  
