#矩阵常用操作
import numpy as np
nd14 = np.arange(9).reshape([3, 3])
#矩阵转置
np.transpose(nd14)
#矩阵乘法运算
a = np.arange(12).reshape([3, 4])
b = np.arange(8).reshape([4, 2])
a.dot(b)
#求矩阵的迹
a.trace()
#计算矩阵行列式
np.linalg.det(nd14)
#计算逆矩阵
c = np.random.random([3, 3])
np.linalg.solve(c, np.eye(3))

'''
diag:以一维数组方式返回方阵对角线元素
dot:矩阵乘法
trace:求迹，即对角线元素的和
det:计算行列式
eig:计算特征值和特征向量
inv:计算方阵的逆
qr:计算qr分解
svd:计算奇异值分解svd
solve:解线性方程组Ax = b,其中A为方阵
lstsq:计算Ax = b的最小二乘解
'''