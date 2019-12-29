import numpy as np
from scipy.sparse.linalg import eigs

def Eigen_Reweighting(X,order,coef): #筛选出来的A的top-L（绝对值从大到小）个特征值对应的F(A)的特征值
# X: original eigenvalues #L个A的特征值
# order: order, -1 stands for infinity #我们考虑从1阶临近到order阶临近的所有阶数，order是一个正整数
# coef: weights, decaying constant if order = -1
# return: reweighted eigenvalues
    if order == -1:     # infinity 次数是无穷
        assert len(coef) == 1, 'Eigen_Reweighting wrong.'
        coef = coef[0]
        assert np.max(np.absolute(X)) * coef < 1, 'Decaying constant too large.'
        X_H = np.divide(X, 1 - coef * X) #F(A)定义为首项是X，公比是coef * X的等比级数
    else:
        assert len(coef) == order, 'Eigen_Reweighting wrong.'
        X_H = coef[0] * X
        X_temp = X
        for i in range(1,order):
            X_temp = np.multiply(X_temp,X)
            X_H += coef[i] * X_temp
    return X_H


def Eigen_TopL(A, d,N): #求A的top-L特征值和特征向量,L和d相关（最小的L，使得top-L个特征值中至少有d个正数）
# A: N x N symmetric sparse adjacency matrix
# d: preset dimension
# return: top-L eigen-decomposition of A containing at least d positive eigenvalues这d个A的特征值对应的F(A)的特征值也是top-L的
#无论对A还是F(A),这里所说的top-L是指特征值绝对值的top-L个
    # assert np.all(A.T == A), 'The matrix is not symmetric!'
    L =d 
    lambd = np.array([0]) #初始化特征值
    #print('N,d=',N,d)
    while sum(lambd > 0) < d:         # can be improved to reduce redundant calculation if L <= 2d + 10 not hold
    #如果lambd中大于0的元素个数小于d
        L=min(L+d,N-1)
        lambd, X = eigs(A, L) #求稀疏矩阵A的前L个特征值,绝对值top-L的特征值？排列顺序为：先正后负，先绝对值大，后绝对值小
        lambd, X = lambd.real, X.real #取实部，实对称矩阵A的特征值都是实数，特征向量都是实向量，所以这步不需要！
        #print('lambd=',lambd)
        #print('L,sum(lambd > 0)=',L,sum(lambd > 0))
        # only select top-L
    #print('final N,L,d=',N,L,d)
    #print('lambd=',lambd)
    temp_index = np.absolute(lambd).argsort()[::-1] #特征值绝对值从大到小排序后依次返回它们在原lamba中的索引
    lambd = lambd[temp_index] #将特征值存为按其绝对值从大到小排顺序，存为lambd 
    temp_max, = np.where(np.cumsum(lambd > 0) >= d) #为什么有个逗号？np.cumsum(lambd > 0)：数lambd前0，1，2,...个元素中有多少个大于0
    #lambd中前temp_max个元素中至少有d个大于0.  temp_max是一维数组，temp_max[0]就是满足条件的最小元素个数-1，也就是说，L=temp_max[0]+1
    lambd, temp_index = lambd[:temp_max[0]+1], temp_index[:temp_max[0]+1] #取绝对值前L=temp_max[0]+1大的特征值，以及它们对应在没按大小排序的lambd中的索引
    X = X[:,temp_index] #取绝对值前L大的特征值对应的特征向量，len(lambd)=len(tem_index)=L
    return lambd, X


def Shift_Embedding(lambd, X, order, coef, d): 
# lambd, X: top-L eigen-decomposition #矩阵A的top-L特征分解
# order: a number indicating the order
# coef: a vector of length order, indicating the weights for each order
# d: preset embedding dimension
# return: content/context embedding vectors
    lambd_H = Eigen_Reweighting(lambd,order,coef)             # High-order transform
    #print('lambd_H=',lambd_H) #筛选出来的L个A的特征值对应的F(A)的特征值
    temp_index = np.absolute(lambd_H).argsort()[::-1]         # 按照删选的F(A)的那些特征值(即：lambd_H)从大到小返回它们在lambd_H中的索引
    temp_index = temp_index[:d+1]   #截取temp_index的前d+1个元素，即按照删选的F(A)的那些top-(d+1)个特征值(即：lambd_H)在lambd_H中的索引
    lambd_H = lambd_H[temp_index]   #这top-(d+1)个特征值的数值
    #print('type(temp_index)',type(temp_index)) #np.array格式
    #print('type(lambd_H)',type(lambd_H)) #np.array格式
    lambd_H_temp = np.sqrt(np.absolute(lambd_H)) #绝对值后再开根号
    U = np.dot(X[:,temp_index], np.diag(lambd_H_temp)) 
    # Calculate embedding, U*=U*Sigma**2,Sigma是筛选的F(A)的特征值组成的对角阵，这里U=X[:,temp_index]       
    V = np.dot(X[:,temp_index], np.diag(np.multiply(lambd_H_temp, np.sign(lambd_H)))) #V*=V*Sigma**2，这里V=X[:,temp_index]*np.sign(lambd_H) 
    return U, V

    
def AROPE(A, d, order, weights,N):
# A: adjacency matrix A or its variations, sparse scipy matrix
# d: dimensionality 
# r different high-order proximity:
    # order: 1 x r vector, order of the proximity
    # weights: 1 x r list, each containing the weights for one high-order proximity
# return: 1 x r list, each containing the embedding vectors 
    A = A.asfptype() #邻接矩阵格式转换为浮点格式的矩阵 
    lambd, X = Eigen_TopL(A, d,N) #求A的top-L（依照绝对值大小）特征值和对应的特征向量，其中含有至少d个正特征值
    #print('lambd=',lambd)       
    r = 1 #r种不同的proximity的取法
    U_output, V_output = [], []
    for i in range(r):
        U_temp, V_temp = Shift_Embedding(lambd, X, order[2], weights[2], d) 
        #order[2]=3，也就是说，我们只考虑1，2，3阶proximity，它们的权重分别是weights[2]里面的元素
        U_output.append(U_temp)
        V_output.append(V_temp)
    return U_output, V_output

