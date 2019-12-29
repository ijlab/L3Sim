# -*- coding: utf-8 -*-
from __future__ import division 
import time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from sklearn.metrics import auc
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_curve
from numpy import interp

#用python2运行！！！
t1=time.time()
print('Network: S.pombe')
iteration=10
top=200

mean_fpr=[i for i in range(1,top+1)]
#mean_fpr=np.array(mean_fpr)
#print('mean_fpr=',mean_fpr)
methodname=['Sim','SimCN','SimPA','SimRA','L3','LO','L3+Sim']
mean_Ys,mean_AUCs=[],[]
for j in range(len(methodname)):
  fpr,tprs,aucs,Pre=[],[],[],[]
  plt.figure(j)
  plt.figure(figsize=(8,6)) #画布大小    score_fold10
  plt.xlim([0,top])
  plt.ylim([0,1])
  for kfold in range(iteration):
    precision=np.loadtxt('//extend2//chenyu//Sim//S.pombe//%sprecision_fold%d.txt'%(methodname[j],kfold+1))  
    precision=[precision[i] for i in range(top)] #前top次预测的precision
    AUC=auc(mean_fpr,precision)
    tprs.append(precision)
    plt.plot(mean_fpr,precision,lw=1,alpha=0.3,label='%s fold %d(area=%0.6f)'%(methodname[j],kfold+1,AUC))
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
  mean_tpr=np.mean(tprs,axis=0)
  #print('%s mean precision of top 100, 200, 300, 400, 500 and 1000='%methodname[j],mean_tpr[99],mean_tpr[199],mean_tpr[299],mean_tpr[399],mean_tpr[499],mean_tpr[599],mean_tpr[999])
  mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
  #std_auc=np.std(y,axis=0)
  mean_Ys.append(mean_tpr)
  mean_AUCs.append(mean_auc)
  plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean (area=%0.6f)'%mean_auc,lw=2,alpha=.8)
  std_tpr=np.std(tprs,axis=0)
  tprs_upper=np.minimum(mean_tpr+std_tpr,1)
  tprs_lower=np.maximum(mean_tpr-std_tpr,0)
  plt.fill_between(mean_fpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
  plt.xlim([0,top])
  plt.ylim([0,1])
  plt.xlabel('Prediction times: k')
  plt.ylabel('Precision')
  plt.title('Network S.pombe: precision per k predictions of %s'%methodname[j])
  plt.legend(loc='best')  
  plt.savefig('//extend2//chenyu//Sim//S.pombe//S.pombe_%s.jpg'%methodname[j])
  plt.show() 


plt.figure(4)
plt.figure(figsize=(8,6)) #画布大小 
plt.xlim([0,top])
plt.ylim([0,1])
plt.xlabel('Prediction times: k')
plt.ylabel('Precision')
plt.title('Network S.pombe: mean precision per k predictions')
plt.plot(mean_fpr,mean_Ys[0],color='b',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[0],mean_AUCs[0]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[1],color='#ff81c0',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[1],mean_AUCs[1]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[2],color='#7e1e9c',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[2],mean_AUCs[2]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[3],color='#411900',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[3],mean_AUCs[3]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[4],color='g',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[4],mean_AUCs[4]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[5],color='y',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[5],mean_AUCs[5]),lw=2,alpha=.8)
plt.plot(mean_fpr,mean_Ys[6],color='r',label=r'%s Mean precision curve (area=%0.6f)'%(methodname[6],mean_AUCs[6]),lw=2,alpha=.8)

plt.legend(loc='best')  
plt.savefig('//extend2//chenyu//Sim//S.pombe//S.pombe.jpg')
plt.show()    


t2=time.time()
print("Total time:%s"%(t2-t1))  