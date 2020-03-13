#对矩阵进行操作
import numpy as np
np.random.seed(2018)
nd11=np.random.random([10])
print(nd11)
print(nd11[3])          #获取第3+1个元素
print(nd11[3:6])        #获取第3+1到第6个元素
print(nd11[1:6:2])      #从第1+1到第6，步长为2获取元素
print(nd11[::2])        #从第1开始，步长为2获取元素
nd12=np.arange(25).reshape([5,5])           #此处reshape将原矩阵按顺序重塑为5行5列矩阵
print((nd12))
print(nd12[1:3,1:3])                        #截取2、3行，2、3列元素
print(nd12[(nd12>3)&(nd12<10)])
print(nd12[[1,2]])                          #此处截取2、3行，原理不明，容易理解的方式是print(nd12[1:3,:])
print(nd12[:,1:3])


'''输出
[0.88234931 0.10432774 0.90700933 0.3063989  0.44640887 0.58998539
 0.8371111  0.69780061 0.80280284 0.10721508]
0.3063988986063515
[0.3063989  0.44640887 0.58998539]
[0.10432774 0.3063989  0.58998539]
[0.88234931 0.90700933 0.44640887 0.8371111  0.80280284]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
[[ 6  7]
 [11 12]]
[4 5 6 7 8 9]
[[ 5  6  7  8  9]
 [10 11 12 13 14]]
[[ 1  2]
 [ 6  7]
 [11 12]
 [16 17]
 [21 22]]
'''