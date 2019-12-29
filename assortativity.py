# -*- coding: utf-8 -*-
from __future__ import division 
import networkx as nx
from copy import deepcopy
import random,json,time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import auc
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
#matplotlib.use("Agg")
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_curve
from scipy.special import comb
import xlwt
t1=time.time()
#用python2运行！！！！
###############################################################################
info=[['graph name','#nodes','#edges','average clustering coefficient','assortativity coefficient','average degree','degree heterogeneity','link density','#rings']]
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//A.thaliana//A.thaliana.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
     
info.append(['A.thaliana',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Arabidopsis1//Arabidopsis1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['Arabidopsis',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//B.subtilis//B.subtilis.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['B.subtilis',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//BIOGRID-PF//BIOGRID-PF.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3) 

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
    
info.append(['BIOGRID-PF',n,m,C,r,k1,k2,D,ring])

########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//BIOGRID-RN//BIOGRID-RN.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['BIOGRID-RN',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//C.elegans//C.elegans.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['C.elegans',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//coli1//coli1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['coli',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//D.Melanogaster//D.Melanogaster.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)   

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
  
info.append(['D.Melanogaster',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//E.coli//E.coli.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['E.coli',n,m,C,r,k1,k2,D,ring])

########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//hi-ii-14//hi-ii-14.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['hi-ii-14',n,m,C,r,k1,k2,D,ring])

########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//hi-iii//hi-iii.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)   

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
  
info.append(['hi-iii',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//hi-tested//hi-tested.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['hi-tested',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Human1//Human1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)    

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
 
info.append(['Human',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//marina1//marina1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['marina',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//mouse1//mouse1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['mouse',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Oryza1//Oryza1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round(m/comb(n,2),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
     
info.append(['Oryza',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//PrePPI-human2011//PrePPI-human2011.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3) 

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
    
info.append(['PrePPI-human2011',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.cerevisiae//S.cerevisiae.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round(m/comb(n,2),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
     
info.append(['S.cerevisiae',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//S.pombe//S.pombe.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['S.pombe',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//yeast1//yeast1.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)    

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
 
info.append(['yeast',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//YeastS//YeastS.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)   

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
  
info.append(['YeastS',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Chicago//Chicago.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round(m/comb(n,2),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
     
info.append(['Chicago',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Bible//Bible.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3) 

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
    
info.append(['Bible',n,m,C,r,k1,k2,D,ring])
########################################################

G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//erdos_renyi_n500_p04//erdos_renyi_n500_p04.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
   
info.append(['erdos_renyi_n500_p04',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//erdos_renyi_n500_p06//erdos_renyi_n500_p06.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round(m/comb(n,2),3)  

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
     
info.append(['erdos_renyi_n500_p06',n,m,C,r,k1,k2,D,ring])
########################################################

G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//erdos_renyi_n500_p08//erdos_renyi_n500_p08.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['erdos_renyi_n500_p08',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//erdos_renyi_n500_p10//erdos_renyi_n500_p10.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['erdos_renyi_n500_p10',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Euroroad//Euroroad.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['Euroroad',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//Infectious//Infectious.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['Infectious',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//netsience//netsience.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['netsience',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//watts_strogatz_n500_k20_p10//watts_strogatz_n500_k20_p10.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['watts_strogatz_n500_k20_p10',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//watts_strogatz_n500_k40_p10//watts_strogatz_n500_k40_p10.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['watts_strogatz_n500_k40_p10',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//watts_strogatz_n500_k60_p10//watts_strogatz_n500_k60_p10.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3)     

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1

info.append(['watts_strogatz_n500_k60_p10',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//watts_strogatz_n500_k80_p10//watts_strogatz_n500_k80_p10.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3) 

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
    
info.append(['watts_strogatz_n500_k80_p10',n,m,C,r,k1,k2,D,ring])
########################################################
G=nx.read_weighted_edgelist('//extend2//chenyu//Sim//arenas-email//arenas-email.txt') #读取原图
n,m=len(G.nodes()),len(G.edges)
C=round(nx.average_clustering(G),3)
r=round(nx.degree_pearson_correlation_coefficient(G),3)
degree=[G.degree(i) for i in G.nodes()]
degree2=[i**2 for i in degree]
k1=np.mean(degree)
k2=np.mean(degree2)/k1**2 
k1=round(k1,3)
k2=round(k2,3)
D=round((2*m)/(n**2+n),3) 

ring=0
for (u,v,w) in G.edges(data=True):
    if u==v:
       ring+=1
    
info.append(['arenas-email',n,m,C,r,k1,k2,D,ring])
########################################################

print (info)
np.savetxt("//extend2//chenyu//Sim//info.txt",info,fmt="%s",delimiter=",")



f = xlwt.Workbook() #创建工作簿
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
for j in range(len(info)):
  for i in range(len(info[0])):
    sheet1.write(j,i,info[j][i])#表格的第一行开始写。第一列，第二列。。。。 
f.save('//extend2//chenyu//Sim//info.xls')#保存文件

t2=time.time()
print("time:%s"%(t2-t1))

