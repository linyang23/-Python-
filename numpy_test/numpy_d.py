# coding=utf-8
#随机数生成
import numpy as np
np.random.seed(123)             #更换随机数序列为第123组，若没有此句则会随机选取一组序列
nd5_1=np.random.randn(2,3)      #randn是产生-1到1，randm是产生0到1
print(nd5_1)
np.random.shuffle(nd5_1)        #shuffle的功能是将序列进行随机排序，目前的观察是两行互换
print(nd5_1)
print(type(nd5_1))

'''输出 
[[-1.0856306   0.99734545  0.2829785 ]
 [-1.50629471 -0.57860025  1.65143654]]
[[-1.50629471 -0.57860025  1.65143654]
 [-1.0856306   0.99734545  0.2829785 ]]
<class 'numpy.ndarray'>
'''