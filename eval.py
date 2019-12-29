
import numpy as np

def Precision_Np(Matrix_test,Matrix_train,U,V,Np):
# Matrix_test is n x n testing matrix, may overlap with Matrix_train
# Matrix_train is n x n training matrix
# U/V are content/context embedding vectors
# Np: returns Precision@Np for pairwise similarity 
    N, _ = U.shape
    assert N < 30000, 'Network too large. Sample suggested.' #如果节点个数小于30000时才进行下一步，否则建议先采样
    Sim = U.dot(V.T) #矩阵U乘以V的转置,结果Sim是N*N方阵，因为U和V都是N*(d+1)矩阵
    negative_elements_num=0
    for i in range(Sim.shape[0]):
        for j in range(Sim.shape[0]):
            if Sim[i,j]<0:
               negative_elements_num+=1   
    print('negative_elements_num in Sim=',negative_elements_num)
    #print('Matrix_train=',Matrix_train)
    #print('Matrix_test=',Matrix_test)
    temp_row, temp_col = np.nonzero(Sim) #返回Sim矩阵中非零元的索引,排除了0元素，留下了正数和负数？？
    temp_value = Sim[temp_row,temp_col] #Sim矩阵的非零元的数值
    #temp_choose = np.logical_and(np.array(Matrix_train[temp_row,temp_col])[0] == 0, 1==1)
    temp_choose = np.logical_and(np.array(Matrix_train[temp_row,temp_col])[0] == 0, temp_row != temp_col)#非主对角线的所有正负样本
    #逻辑与，返回true或者false，布尔格式，当Sim的非零元对应于训练矩阵中位置的元素为0，且不是对角元素的话，返回true。排除了主对角线元素？？
    #考虑在训练矩阵中是0的位置，是因为需要挖掘训练矩阵中的0元素为潜在的非零元（将训练网络中没有的边链接的节点对，预测它们之间有边），如果训练矩阵取零矩阵，且图嵌入的输入是网络，那么就是还原网络的问题
    #print('temp_choose=',temp_choose)
    temp_row, temp_col, temp_value = temp_row[temp_choose], temp_col[temp_choose], temp_value[temp_choose]#非主对角线的所有正负样本的坐标和Sim数值
    #选取训练矩阵中是0而在Sim中非0的位置的元素在Sim矩阵中的行标、列标以及它的数值（成对应关系）。
    #print('temp_row, temp_col, temp_value=',temp_row, temp_col, temp_value)
    temp_index = np.argsort(temp_value)[::-1] #数值按从大到小排列，并返回其在temp_value中的索引
    assert len(temp_index) >= Np, 'Np too large'  #如果Sim中非零元的个数大于Np时才进行下一步，否则打印：Np太大了
    temp_index = temp_index[: int(Np)+1] #截断取Sim中非零元前Np大的那些元素在temp_value中的索引（和在temp_row和 temp_col中的索引一样）
    temp_value=list(temp_value)
    temp_value.sort(reverse=True)
    temp_row, temp_col = temp_row[temp_index], temp_col[temp_index] #截断取Sim中非零元前int(Np)+1大的那些元素在Sim矩阵中的行标和列标
    '''print('Sim[1,3]=',Sim[1,3])
    print('temp_row[0], temp_col[0],temp_value[0]=',temp_row[0], temp_col[0],temp_value[0])
    print('temp_row[1], temp_col[1]=,temp_value[1]',temp_row[1], temp_col[1],temp_value[1])
    print('temp_row[2], temp_col[2]=,temp_value[2]',temp_row[2], temp_col[2],temp_value[2])'''
    #print('temp_value=',temp_value)
    result = np.array(Matrix_test[temp_row,temp_col])[0] > 0 #在测试集中这些位置的元素大于0吗，如果是就返回true
    result = np.divide(np.cumsum(result > 0), np.array(range(len(result))) + 1) 
    #np.cumsum(result > 0)是result中前i个元素中true的个数（真阳性个数），i=1,2,...,len(result);  result中的元素是true或者false，也就是1或者0
    #np.array(range(len(result))) + 1是[1,2,...,len(result)],也就是预测次数
    #两者相除即为[1,2,...,len(result)]次预测的准确率
    return result