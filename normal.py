# -*- coding: utf-8 -*-
from __future__ import division
import networkx as nx
import time
t1=time.time()
g=nx.read_weighted_edgelist('//extend2//chenyu//Sim//netsience//netsience.txt', delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8') 
weight=[]
for (i,j,d) in g.edges(data=True):
    weight.append(d['weight'])
mi,ma=min(weight),max(weight) #最小，最大权   
for (i,j,d) in g.edges(data=True):
    d['weight']=(d['weight'])/ma  #边权归一化
nx.write_weighted_edgelist(g,'//extend2//chenyu//Sim//netsience//netsience.txt')     

t6=time.time()
print('Overall time:%s'%(t6-t1)) 