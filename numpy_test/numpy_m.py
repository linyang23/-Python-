# coding=utf-8
#广播机制,其功能为方便不同shape的数组（NumPy库的核心数据结构）进行数学运算

import numpy as np
a = np.arange(10)
b = np.arange(10)
#两个shape相同的数组相加
print(a + b)
#一个数组与标量相加
print(a + 3)
#两个向量相乘
print(a * b)
#多维数组之间的运算
c = np.arange(10).reshape([5, 2])
d = np.arange(2).reshape([1, 2])
#首先将d数组进行复制扩充为[5, 2]，然后相加
print(c + d)

'''
输出：
[ 0  2  4  6  8 10 12 14 16 18]
[ 3  4  5  6  7  8  9 10 11 12]
[ 0  1  4  9 16 25 36 49 64 81]
[[ 0  2]
 [ 2  4]
 [ 4  6]
 [ 6  8]
 [ 8 10]]
'''