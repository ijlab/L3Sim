xxx是文件名
xxx.txt 是PPI网络，格式是： 蛋白1 蛋白2 置信度
xxx.fa 是PPI网络中蛋白的序列，fa标准格式，一行蛋白名称，一行序列
txt和fa文件均来自于STRING数据库

linux上用xshell操作的命令（后台挂起运行）：

nohup python -u alig.py xxx.fa aligxxx.txt > aligxxx.log 2>&1 &
nohup python3 -u jacc.py xxx.txt > jacc.log 2>&1 &

nohup python -u alig.py 30.fa alig30.txt > alig.log 2>&1 &
nohup python3 -u jacc.py 30 > jacc.log 2>&1 &

alig.py是修改以后的计算序列比对的得分算法，需要调用同一层文件夹里面的aligment.py
jacc.py是计算PPI网络的杰卡德相似度，并按从大到小排序
----------------------------------------
1 Arabidopsis thaliana
number of nodes:
40
number of edges:
129
average node degree:
6.45
avg. local clustering coefficient:
0.625
蛋白1 蛋白2 杰卡德相似度  序列比对得分
PIN7 PIN3 0.7913834 0.842024539877
----------------------------------------
2 Caenorhabditis elegans
number of nodes:
41
number of edges:
205
average node degree:
10
avg. local clustering coefficient:
0.713
蛋白1 蛋白2 杰卡德相似度  序列比对得分
abu-7 abu-6 0.70384985 0.874429223744
----------------------------------------
30 Escherichia coli K12 MG1655  
number of nodes:
35
number of edges:
87
average node degree:
4.97
avg. local clustering coefficient:
0.598
蛋白1 蛋白2 杰卡德相似度  序列比对得分
rhsA rhsB 1.0 0.896167247387
rhsD rhsA 0.8021216 0.711538461538
rhsD rhsB 0.7920696 0.746896551724
----------------------------------------
400 Mus musculus
number of nodes:
201
number of edges:
2075
average node degree:
20.6
avg. local clustering coefficient:
0.648
蛋白1 蛋白2 杰卡德相似度  序列比对得分
Camk2a Camk2g 0.8977933 0.766917293233
Camk2g Camk2d 0.91231686 0.833955223881
Camk2a Camk2d 0.9082842 0.829457364341
Camk2b Camk2g 0.9142654 0.827648114901
Camk2a Camk2b 0.9103021 0.765567765568
Camk2b Camk2d 0.9255441 0.834545454545
----------------------------------------
50 Homo sapiens  
number of nodes:
101
number of edges:
1424
average node degree:
28.2
avg. local clustering coefficient:
0.658
蛋白1 蛋白2 杰卡德相似度  序列比对得分 
RAB5B RAB5C 0.9483445 0.748
RAB5A RAB5C 0.78506744 0.764940239044
RAB5A RAB5B 0.7846903 0.819004524887
----------------------------------------
60 Saccharomyces cerevisiae
number of nodes:
101
number of edges:
1077
average node degree:
21.3
avg. local clustering coefficient:
0.675
蛋白1 蛋白2 杰卡德相似度  序列比对得分 
TDH2 TDH1 0.87267643 0.885542168675
TDH3 TDH2 0.8513397 0.963855421687
TDH3 TDH1 0.81701237 0.882530120482
----------------------------------------
10 Drosophila melanogaster