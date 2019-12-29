# -*- coding: utf-8 -*-
from __future__ import division 
import networkx as nx
import time
from copy import deepcopy
###############################################################################
t1=time.time()
it=.5  #设定最小权重，删去小于这个数值的边
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//511145new//511145.txt') #读取原图
print ('Info of Network: 511145')
print (nx.info(G))
H=deepcopy(G)
for (i,j,d) in G.edges(data=True):
    if d['weight']<it:
       H.remove_edge(i,j)  
nx.write_weighted_edgelist(H,'//extend2//chenyu//Sim//511145new//511145new.txt')   
H=nx.read_weighted_edgelist('//extend2//chenyu//Sim//511145new//511145new.txt') #读取原图 
print ('Info of Network: 511145new (deleting links with weak weights(<%f))'%it)
print (nx.info(H))    
t2=time.time()
print("time:%s"%(t2-t1))  