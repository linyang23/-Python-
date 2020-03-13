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