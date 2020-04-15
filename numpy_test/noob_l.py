#通用函数

#使用math与numpy函数性能比较
import time
import math
import numpy as np
x = [i * 0.001 for i in np.arange(1000000)]
start = time.clock()
for i, t in enumerate(x):
    x[i]=math.sin(t)
print("math.sin", time.clock() - start)
x = [i * 0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.clock()
np.sin(x)
print("numpy.sin:", time.clock() - start)
'''
输出：
math.sin 0.7915972799999995
numpy.sin: 0.023284486999997966
'''
#由此可见，numpy.sin比math.sin快近40倍

#使用循环与向量运算比较
x1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
#使用循环计算向量点积
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot = " + str(dot) + "\n for loop----- Computation time = " + str(1000 * (toc - tic)) + "ms")
#使用numpy函数求点积
tic = time.process_time()
dot = 0
for i in range(100):        #由于时间间隔过短导致toc和tic之间存储精度不够，所以此处循环运行100遍扩大差距
    dot = np.dot(x1, x2)
toc = time.process_time()
print("dot = " + str(dot) + "\n vector version----- Computation time = " + str(10 * (toc - tic)) + "ms")        #由于上面循环了100遍，所以此处只乘以10而非1000

'''
输出：
dot = 250518.43859786313
 for loop----- Computation time = 1062.5ms
dot = 250518.43859785725
 vector version----- Computation time = 2.5ms
'''
#从程序运行结果上来看，该例子使用for循环的运行时间是使用向量运算的运行时间的约400倍。因此，深度学习算法中，一般都使用向量化矩阵运算

'''
常用通用函数
sqrt:计算序列化数据的平方根
sin,cos:三角函数
abs:计算序列化数据的绝对值
dot:矩阵运算
log,log10,log2:对数运算
exp:指数运算
cumsum，cumproduct:累计求和，求积
sum:对一个序列化数据进行求和
mean:计算均值
median:计算中位数
std:计算标准差
var:计算方差
corrcoef:计算相关系数
'''