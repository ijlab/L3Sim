import numpy as np
import alignment as ag
import sys
import time
from concurrent.futures import ThreadPoolExecutor as TPE
file_p=open(sys.argv[1],'r')

outfile=open(sys.argv[2],'w')

file_read=file_p.read()
lines=file_read.split('\n')

namelist=[]
seqlist=[]
def add_list(item):
  if lines.index(item)%2==0:
    name_s=item.split('\t')[0]
    cname=name_s.split('>')[1]
    namelist.append(cname)
  else:
    seqlist.append(item)

with TPE(16) as exec:
  exec.map(add_list,lines)

print (len(namelist),'proteins')
oldt=time.time()
#print(seqlist)
seqlen=len(namelist)
q=np.zeros(seqlen)
#'''
def one2all(item):
  nameid=namelist.index(item)
 # print(nameid,seqlen)
  for i in range(nameid+1,seqlen):
  #  print(i)
    name2=namelist[i]
    seq2=seqlist[i]
    seq1=seqlist[nameid]
    strt=item+' '+name2+' '+str(ag.needle(seq1,seq2)/100)
    print(strt)
    q[namelist.index(item)]+=1
    outfile.write(strt)
    outfile.write('\n')
    outfile.flush()

with TPE(32) as exec2:
  for each in namelist:
    exec2.submit(one2all,each)
'''
for i in range(seqlen):
  for j in range(i+1,seqlen):
    seq1=seqlist[i]
    seq2=seqlist[j]
    name1=namelist[i]
    name2=namelist[j]
    strt=name1+' '+name2+' '+str(ag.needle(seq1,seq2)/100)
    print(strt)
    outfile.write(strt)
    outfile.write('\n')
    outfile.flush()

outfile.close()
'''

print('timecost : ',1000*(time.time()-oldt)/1000 )
