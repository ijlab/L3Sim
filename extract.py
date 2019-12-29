# -*- coding: utf-8 -*-
import numpy as np
import time
import re
#用python2运行！！！
t1=time.time()
def itercolumn23(filename, splitregex = '\t'):
    with open(filename, 'rt') as handle:
        for ln in handle:
            items = re.split(splitregex, ln)
            yield items[0], items[1]
b=[] 
for x, y in itercolumn23("//extend2//chenyu//Sim//Infectious//Infectious.txt", splitregex='\s+'):
    b.append([x,y,1])
np.savetxt("//extend2//chenyu//Sim//Infectious//Infectious.txt", b, fmt='%s')

t2=time.time()
print("Total time:%s"%(t2-t1))  