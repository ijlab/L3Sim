# -*- coding: utf-8 -*-
from __future__ import division 
import networkx as nx
from copy import deepcopy
import random,time

t1=time.time()
###############################################################################
print ('Network: ego-facebook')
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//ego-facebook//ego-facebook.txt') #读取原图

for (u, v, d) in G.edges(data=True): 
    d['weight']=1     
###############################################################################
print ('nx.info(G)')
print (nx.info(G))
#print ('average clustering coefficient:', nx.average_clustering(G))
nx.write_weighted_edgelist(G, '//extend2//chenyu//Sim//ego-facebook//ego-facebook.txt')#保存原图
print ('weight phase is done')
t2=time.time()
print("time:%s"%(t2-t1))  
