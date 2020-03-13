#矩阵生成
import numpy as np
nd6=np.zeros([3,3])     #zeros，生成纯0矩阵
nd7=np.ones([3,3])      #ones，生成纯1矩阵
nd8=np.eye(3)           #生成对角线纯1，其余为0的对角矩阵
print(nd6)
print(nd7)
print(nd8)
print(np.diag([1,2,3])) #生成对角为1，2，3的对角矩阵并输出