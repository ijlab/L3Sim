import alignment
import sys
import time
t1=time.time()

file1=open(sys.argv[1],'r')
file2=open(sys.argv[1],'r')

full1=file1.read()
lines1=full1.split('\n')
len1=len(lines1)

full2=file2.read()
lines2=full2.split('\n')
len2=len(lines2)

outfile=open(sys.argv[2],'w')

for i in range((len1-1)/2):
  name1_s=lines1[i*2].split('\t')[0]
  name1=name1_s.split('>')[1]
  seq1=lines1[i*2+1]
  for j in range(i+1,(len2-1)/2):
    name2_s=lines2[j*2].split('\t')[0]
    name2=name2_s.split('>')[1]
    seq2=lines2[j*2+1]
    strt=name1+' '+name2+' '+str(alignment.needle(seq1,seq2)/100)
    print(strt)
    outfile.write(strt)
    outfile.write('\n')
    outfile.flush()
outfile.close()

t2=time.time()
print("Total time:%s"%(t2-t1))  

